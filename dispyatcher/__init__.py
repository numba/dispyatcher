import copy
import ctypes
import weakref
from typing import Sequence, List, Dict, Union, TypeVar, Generic, Callable

import llvmlite.binding
import llvmlite.ir
from llvmlite.ir import Type as LLType, IRBuilder, Value as IRValue, PointerType


def llvm_type_to_ctype(ty: LLType):
    """
    Find a ctype representation from an LLVM type

    This function finds the appropriate ctype for an LLVM type, if one exists. The ctype representation may lose
    information about the LLVM type, especially where structures are concerned. Some LLVM types do not have equivalent
    ctype representations including half-precision floats (aka `float16`), vectors, metadata types, and label types.

    If conversion fails, an exception is raised.

    A LLVM type with no corresponding ctype can still be used as long as it is not in the signature of a call site.
    It is permissible to have types with no ctype-equivalent as intermediate values or laundered through opaque data
    structures.
    """
    if isinstance(ty, llvmlite.ir.VoidType):
        return None
    if isinstance(ty, llvmlite.ir.IntType):
        if ty.width == 8:
            return ctypes.c_int8
        if ty.width == 16:
            return ctypes.c_int16
        if ty.width == 32:
            return ctypes.c_int32
        if ty.width == 64:
            return ctypes.c_int64
    if isinstance(ty, llvmlite.ir.HalfType):
        raise TypeError("Half types have no ctype representation")
    if isinstance(ty, llvmlite.ir.FloatType):
        return ctypes.c_float
    if isinstance(ty, llvmlite.ir.DoubleType):
        return ctypes.c_double
    if isinstance(ty, llvmlite.ir.PointerType):
        if (isinstance(ty.pointee, llvmlite.ir.VoidType) or
           isinstance(ty.pointee, llvmlite.ir.BaseStructType) and not ty.pointee.elements):
            return ctypes.c_void_p
        return ctypes.POINTER(llvm_type_to_ctype(ty.pointee))
    if isinstance(ty, llvmlite.ir.ArrayType):
        return llvm_type_to_ctype(ty.element) * ty.count
    if isinstance(ty, llvmlite.ir.BaseStructType):
        class ConvertedStructure(ctypes.Structure):
            _fields_ = [(name, llvm_type_to_ctype(field_type)) for name, field_type in ty.elements]
        return ConvertedStructure
    raise TypeError(f"No conversion from {ty} to ctypes")


def is_llvm_floating_point(ty: llvmlite.ir.Type) -> bool:
    """
    Checks if the LLVM type provided is a floating point type
    :param ty: the LLVM type to check
    :return: true if an instance of `llvm.ir.HalfType`, `llvm.ir.FloatType`, or `llvm.ir.DoubleType`
    """
    return (isinstance(ty, llvmlite.ir.HalfType) or isinstance(ty, llvmlite.ir.FloatType) or
            isinstance(ty, llvmlite.ir.DoubleType))


class Type:
    """
    The representation of a parameter or return type that a handle can use
    """
    def into_type(self, target) -> Union["Handle", None]:
        """
        Convert from self type into the target type provided.

        If a conversion is known, this function should return a handle with a signature that has a single parameter of
        its own type and a return type of the target provided. If no suitable conversion exists, the method should
        return `None`.

        This method operates in conjunction with `from_type` to allow either the source or destination type to provide
        a conversion, with the source type having priority.
        """
        pass

    def from_type(self, source) -> Union["Handle", None]:
        """
        Convert from the provided type into the self type.

        If a conversion is known, this function should return a handle with a signature that has a single parameter of
        the source type provided and a return its own type. If no suitable conversion exists, the method should return
        `None`.

        This method operates in conjunction with `into_type` to allow either the source or destination type to provide
        a conversion, with the source type having priority.
        """

        pass

    def ctypes_type(self):
        """
        Provide the ctype representation of this type

        If no ctype representation exists, it should raise a type error. The default method for this method will return
        the ctype representation of the machine/LLVM type. It is provided in the case where a better ctype
        representation is available.
        """
        return llvm_type_to_ctype(self.machine_type())

    def machine_type(self) -> LLType:
        """
        Provide the LLVM/machine representation of the type

        This library will only ever convert from higher-level types to machine types and never back. The reason the two
        type systems are separate is that the conversions that might be technically legal at the machine level might be
        bad ideas. For instance, a high-level type could represent an array with static size information, represented in
        machine type as a pointer, and converting an arbitrary pointer of the same type could be unsafe.

        Therefore, the contract provided is that manipulations will happen only ever on high-level types and the
        machine type is used purely for generating code.
        """
        pass

    def as_pointer(self) -> "Type":
        """
        Creates a new type that is a simple pointer to this type
        :return: the new type
        """
        return Pointer(self)


class Deref(Type):
    """
    A type that can be dereferenced using the automatic dereference operations.
    """
    def target(self) -> Type:
        """
        The type "inside" this type.
        :return: the inner type
        """
        pass


class Pointer(Deref):
    """
    A type for a simple pointer

    This, at the machine level, looks like a C++ `&` reference or a Rust reference. Unlike a C `*` pointer, no
    arithmetic can be performed on it.
    """
    __inner: Type

    def __init__(self, inner: Type):
        self.__inner = inner

    def ctypes_type(self):
        return ctypes.POINTER(self.__inner.ctypes_type())

    def machine_type(self) -> LLType:
        return self.__inner.machine_type().as_pointer()

    def target(self) -> Type:
        return self.__inner

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Pointer):
            return o.__inner == self.__inner
        return False

    def __str__(self) -> str:
        return f"Pointer to {self.__inner}"


class ControlFlow:
    """
    A non-linear flow control mechanism

    This super type can be used for a control flow that does not support alternate control flow. There is a separate
    control flow that allow using the CPython exception mechanism for non-linear flow. The callsite must have one
    top-level flow control, but the handles within don't have to match exactly. It is possible to have handles that can
    adapt from one flow control to another (_e.g._, check `errno` and turn it into a Python exception) or if flow
    controls are logical subsets of one another (_e.g._, a handle using this flow control, which is infallible/linear,
    can be used inside a callsite that has takes another flow control); said another way, you can always use a handle
    that doesn't throw in a callsite that handles a throw.
    """
    __builder: IRBuilder
    __global_addresses: Dict[str, ctypes.c_char_p]
    __cleanup: List[Callable[[IRBuilder], Dict[str, ctypes.c_char_p]]]

    def __init__(self, builder: IRBuilder):
        self.__builder = builder
        self.__global_addresses = {}
        self.__cleanup = []

    @property
    def builder(self) -> IRBuilder:
        """
        Access the LLVM IR builder for this flow control.
        :return:
        """
        return self.__builder

    def defer_cleanup(self, cleanup: Callable[[IRBuilder], Dict[str, ctypes.c_char_p]]) -> None:
        """
        Add an operation that should be done during all cleanup paths.

        Callbacks are placed on a LIFO stack. Whenever the flow control needs to return, it will call all of these
        callbacks to do resource deallocation required. The callbacks do *not* have access to the advanced features of
        the flow control (_i.e._, they cannot throw; they must be infallible).

        Because the flow control may have multiple execution paths, these callbacks must be safe and deterministic to
        call repeatedly. The return (happy) path will also call the cleanup callbacks.

        :param cleanup: this callback will be executed on all exit paths
        """
        self.__cleanup.append(cleanup)

    def _cleanup(self) -> None:
        """
        Perform cleanup on whatever block the builder is currently set to.

        This is meant to allow subclass to perform appropriate cleanup during split flow control. This does not generate
        a return instruction. The caller must terminate the active block. Since all resources are deallocated, no other
        operations should be performed in the block.
        """
        for cleanup in reversed(self.__cleanup):
            self.__global_addresses.update(cleanup(self.builder))

    def finish(self) -> Dict[str, ctypes.c_char_p]:
        """
        Invoked by the callsite to perform any cleanup tasks before it generating the final return instruction and to
        collect any global addresses (exteral bindings) that will need to be injected into the call site.

        If a subclass has additional finalization operations, it should override this method, perform the cleanup, and
        then return the call to the super method.
        """
        self._cleanup()
        return self.__global_addresses

    def upsert_global_binding(self, name: str, ty: LLType, address: ctypes.c_char_p) -> IRValue:
        """
        Create a binding for a value.

        The handle can reference an external function or other value by generating a constant and then stuffing the
        desired address for that constant into this table. LLVM does not permit writing raw addresses for function
        pointers, so this acts as a workaround. Moreover, these addresses can be updated dynamically using the
        invalidation mechanism. Strictly, these don't have to be function pointers; they can be any kind of pointer.
        :param name: the name to use; there is no duplication control, so if two handles use a colliding name for
        different addresses, the behaviour is undefined.
        :param ty: the LLVM type for the symbol; the returned value is a pointer to this type
        :param address: the real machine address to use for this symbol
        :return: an LLVM global constant for this binding
        """
        if name in self.__builder.module.globals:
            return self.__builder.module.globals[name]
        else:
            value = llvmlite.ir.GlobalVariable(self.__builder.module, ty.as_pointer(), name)
            value.initializer = llvmlite.ir.Constant(ty.as_pointer(), llvmlite.ir.Undefined)
            self.__global_addresses[name] = address
            return value

    def use_native_function(self, name: str, ret_ty: LLType, args: Sequence[LLType], var_arg: bool = False) -> IRValue:
        """
        Create a binding for a function exported form the current binary.

        :param name: the name to use
        :param ret_ty: the LLVM type for the return of the function
        :param args: the LLVM type for the return of the function
        :param var_arg: whether the function takes variadic arguments
        :return: an LLVM global constant for this binding
        """
        if name in self.__builder.module.globals:
            return self.__builder.module.globals[name]
        else:
            value = llvmlite.ir.Function(self.__builder.module,
                                         llvmlite.ir.FunctionType(ret_ty, args, var_arg=var_arg),
                                         name)
            return value

    def use_native_global(self, name: str, ty: LLType) -> IRValue:
        """
        Create a binding for a symbol exported form the current binary.

        :param name: the name to use
        :param ty: the LLVM type for the symbol
        :return: an LLVM global constant for this binding
        """
        if name in self.__builder.module.globals:
            return self.__builder.module.globals[name]
        else:
            value = llvmlite.ir.GlobalVariable(self.__builder.module, ty, name)
            return value

    def extend_global_bindings(self, addresses: Dict[str, ctypes.c_char_p]) -> None:
        """
        Directly adds addresses to the global binding pool

        This assumes the constants have already been created in LLVM. This is intended to be used when adapting one
        control flow to another. The constants from the `finish()` method of the inner flow can be added here to the
        outer flow.
        :param addresses: the dictionary of addresses to add
        """
        self.__global_addresses.update(addresses)

    def call(self, handle: "Handle", args: Sequence[IRValue]) -> IRValue:
        """
        Calls another handle.

        This is the correct way to invoke another handle in case the control flow needs to perform some boxing/unboxing
        behaviour. For instance, if a control flow were to handle Rust's `Result` type, this method should be overridden
        in a way that boxes and unboxes the `Result`. In Rust speak, this ensure any function call is suffixed with the
        `?` operator.
        :param handle: the handle to call
        :param args: the arguments to pass to that handle
        :return: the return value from that handle
        """
        return handle.generate_ir(self, args)


class InvalidationListener:
    """
    A mixin for processing update events

    Some handles are stateful and allow runtime mutation. During that mutation, it becomes necessary to recompile any
    call sites that are using the handle. The invalidation mechanism allows those mutations to trigger recompilation of
    dependant call sites through the chain of uses. All consumers of a handle or call site, especially other handles,
    should use the `register` and `unregister` methods to keep updated.
    """

    def invalidate(self) -> None:
        """
        Trigger complete recompilation of any call sites

        Stateful mutable handles will call this method when they have changed in a way that requires recompilation.
        """
        pass

    def invalidate_address(self, name: str, address: ctypes.c_char_p) -> None:
        """
        Trigger updating the address of a call site.

        Using a call site as a handle does not require recompiling any call sites using it, but the function pointer
        that references the updated call site. This indicates that the function pointer has change and needs to be
        updated.
        :param name: the internal name of the call site being used as a handle
        :param address: the new address of the function to use
        """
        pass


class InvalidationTarget(InvalidationListener):
    """
    A mixin for tracking handle updates
    """
    __listeners: weakref.WeakSet[InvalidationListener]

    def __init__(self):
        self.__listeners = weakref.WeakSet()

    def invalidate(self) -> None:
        """
        Trigger invalidation of this object and propagate that to its listeners.
        """
        for listener in self.__listeners:
            listener.invalidate()

    def invalidate_address(self, name: str, address: ctypes.c_char_p) -> None:
        """
        Trigger invalidation of the address of a call site and propagate that to its listeners.
        :param name: the internal name of the call site being used as a handle
        :param address:  the new address of the function to use
        """
        for listener in self.__listeners:
            listener.invalidate_address(name, address)

    def register(self, listener: InvalidationListener) -> None:
        """
        Add a listener to receive invalidation updates for this object
        :param listener: the listener to notify when changes happen
        """
        self.__listeners.add(listener)

    def unregister(self, listener: InvalidationListener) -> None:
        """
        Remove a listener to stop receiving invalidation updates
        :param listener:  the listener to remove
        """
        self.__listeners.remove(listener)


F = TypeVar("F", bound=ControlFlow)


class Handle(InvalidationTarget, Generic[F]):
    """
    A handle is a small function-like behaviour that can be composed and converted to machine code

    Handles, based on the Java Virtual Machine's `MethodHandle`, are a way to dynamically change a kind of special
    function pointer. Every handle is defined by a type signature with a single return value and multiple parameter
    types. Handles can be combined to create new handles with different type signatures. Once inserted into a call site,
    the handles will be converted to machine code and can be executed. The call site can be used as a handle, though it
    serves as a compilation break, where other call sites do not need to be updated when a dependant call site is
    updated.

    Although handles sound like a general purpose intermediate representation that could be used by a compiler, they
    differ in two important ways: handles have intentionally restricted control flow and handles have no shared state.
    In a normal compiler, there are complicated control flows, especially for loops. Handles intentionally do not have
    this kind of complex flow control. Internally, a handle might encapsulate a complicated flow control, but that is
    intentionally not visible to users of that handle. This also relates to the lack of shared state. In a normal
    compiler, it would be typical to allocate a variable and then read and mutate that inside other flow structures,
    such as loop bodies. Handles should not have local effects outside their generated code. Having handles access
    global state is acceptable, if explicitly desired.
    """

    def __add__(self, other):
        if isinstance(other, Handle):
            return PreprocessArgumentHandle(other, 0, self)
        elif callable(other):
            return PreprocessArgumentHandle(other(self.function_type()[0]), 0, self)
        elif isinstance(other, tuple) and callable(other[0]):
            return PreprocessArgumentHandle(other[0](self.function_type()[0], *other[1:]), 0, self)
        else:
            raise TypeError(f"Unsupported operand type for +: {type(other)}")

    def cast(self, target_return_type: Type, *target_parameter_types: Type) -> "Handle":
        """
        Determine the automatic type-directed conversions necessary to convert a handle to the signature provided.

        The number of arguments must match and then automatic type conversion done on each argument individually and
        the return types. Casts are logically distinct from other conversion operations because they don't combine or
        separate arguments.
        If target type is the same as the original type, no conversion is applied.
        :param target_return_type: the new return type
        :param target_parameter_types: the new argument types
        :return: the new handle
        """
        (source_ret, source_args) = self.function_type()
        if source_ret == target_return_type and source_args == target_parameter_types:
            return self

        if len(source_args) != len(target_parameter_types):
            raise TypeError(
                f"Cannot cast {self} as argument counts are ({len(source_args)} → {len(target_parameter_types)}.")

        ret_handle = IdentityHandle(source_ret) if source_ret == target_return_type else (
                source_ret.into_type(target_return_type) or target_return_type.from_type(source_ret))
        if not ret_handle:
            raise TypeError(f"No conversion from {source_ret} to {target_return_type} in cast of {self}.")

        arg_handles = []
        for index in range(0, len(source_args)):
            src_arg = source_args[index]
            tgt_arg = target_parameter_types[index]
            arg_handle = IdentityHandle(src_arg) if src_arg == tgt_arg else (src_arg.from_type(tgt_arg) or
                                                                             tgt_arg.into_type(src_arg))
            if not arg_handle:
                raise TypeError(f"No conversion from {src_arg} to {tgt_arg} at {index} in of {self}.")
            arg_handles.append(arg_handle)

        return CastHandle(self, ret_handle, arg_handles)

    def function_type(self) -> (Type, Sequence[Type]):
        """
        Gets the type of the handle as the return type and a sequence of argument types
        :return: the return type and a sequence of argument types
        """
        pass

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        """
        Convert the handle into LLVM machine code
        :param flow: the flow control builder
        :param args: the arguments to the handle

        :return: the value that is the output of the handle
        """
        pass

    def deref(self, target_type: Union[Type, None] = None) -> "Handle":
        """
        Generate a new handle that calls this handle, whose output must be a pointer, dereferences it and returns that
        value.

        See the `DerefPointer` handle for details on how the typing work.
        :return: the new handle
        """
        (ret_type, _) = self.function_type()
        if not isinstance(ret_type, Deref):
            raise TypeError(f"Don't know how to dereference f{ret_type}")
        return self + DerefPointer(ret_type, target_type)


_call_site_id = 0


def _next_call_site_id() -> str:
    global _call_site_id
    _call_site_id += 1
    return f"call_site_{_call_site_id}"


class ControlFlowType(Generic[F]):
    """
    A manager that allows creation of a new flow control

    In general, a flow control does not require initialization parameters, so these are effectively singletons. A
    callsite will retrain a control flow type, so that each time it needs to rebuild itself, it can initialize the
    control flow.
    """

    def create_flow(self, builder: IRBuilder, return_type: Type, arg_types: Sequence[Type]) -> F:
        """
        Constructs a new control flow for a single compilation pass
        :param builder: the LLVM IR builder to use for compilation
        :param return_type: the return type of the callsite
        :param arg_types: the argument types of the callsite
        :return: the control flow created
        """
        return ControlFlow(builder)

    def ctypes_function(self, return_type: Type, arg_types: Sequence[Type]):
        """
        Creates an appropriate ctypes function type for the call site.

        This can create any synthetic parameters (_e.g._, callbacks for asynchronous flow) or register the calling
        convention with respect to the Python GIL.
        :param return_type: the return type of the callsite
        :param arg_types: the argument types of the callsite
        :return: the function type
        """
        return ctypes.CFUNCTYPE(return_type.ctypes_type(), *(a.ctypes_type() for a in arg_types))


llvmlite.binding.initialize()
llvmlite.binding.initialize_native_target()
llvmlite.binding.initialize_native_asmprinter()

_lljit = llvmlite.binding.create_lljit_compiler()


class CallSite(Handle):
    """
    A callsite is a wrapper around a handle that allows that handle to be compiled and called from Python.

    Callsites can also be used as handles in other callsites and independently updated.
    """
    __ctype: ctypes.CFUNCTYPE
    __engine: Union[llvmlite.binding.ResourceTracker, None]
    __epoch: int
    __flow_type: ControlFlowType
    __handle: Handle[F]
    __id: str
    __other_sites: Dict[str, ctypes.c_char_p]
    __type: llvmlite.ir.FunctionType
    __address: ctypes.c_char_p
    llvm_ir: str

    def __init__(self, handle: Handle[F], flow_type: ControlFlowType = ControlFlowType()):
        """
        Create a new callsite that wraps an existing handle
        :param handle: the handle to place in the callsite; it can be updated later, but a handle must be supplied
        :param flow_type: the control flow to be used for the callsite; the handle must be compatible with this flow
        """
        super().__init__()
        self.__id = _next_call_site_id()
        self.__epoch = 0
        (ret_type, arg_types) = handle.function_type()
        self.__type = llvmlite.ir.FunctionType(ret_type.machine_type(), (a.machine_type() for a in arg_types))
        self.__ctype = flow_type.ctypes_function(ret_type, arg_types)
        self.__flow_type = flow_type
        self.__handle = handle
        self.__engine = None
        self.__cfunc = None
        self.llvm_ir = "; Not yet compiled"
        handle.register(self)
        self.invalidate()

    @property
    def address(self) -> ctypes.c_char_p:
        """
        The address of the compiled version of the callsite (_i.e._, the function pointer that references inside the
        callsite.

        Note that this is not guaranteed to be a stable address. If the callsite is regenerated, this address may
        change. To track the address, use the invalidation subscription mechanism.
        :return: the address to the compiled contents of the callsite
        """
        return self.__address

    @property
    def cfunc(self):
        """
        The Python-callable function for the contents of the call site
        :return:
        """
        return self.__cfunc

    @property
    def handle(self) -> Handle[F]:
        """
        The current handle inside the call site
        :return: the handle
        """
        return self.__handle

    @handle.setter
    def handle(self, handle: Handle[F]):
        assert handle.function_type() == self.__handle.function_type(), "Handle does not match call site signature."
        self.__handle.unregister(self)
        handle.register(self)
        self.__handle = handle
        self.invalidate()

    def invalidate(self):
        # Do not call super as any listeners should not have to update if using a call site as a method handle
        machine_triple = llvmlite.binding.Target.from_default_triple()
        module = llvmlite.ir.Module()
        module.triple = machine_triple.triple
        function = llvmlite.ir.Function(module, self.__type, "call_site")
        builder = IRBuilder(function.append_basic_block())
        (return_type, arg_types) = self.__handle.function_type()
        flow = self.__flow_type.create_flow(builder, return_type, arg_types)
        result = flow.call(self.__handle, function.args)
        global_addresses = flow.finish()
        builder.ret(result)
        module.functions.append(function)

        self.llvm_ir = str(module)
        print(self.llvm_ir)
        print(global_addresses)

        self.__epoch += 1
        builder = llvmlite.binding.JITLibraryBuilder().add_ir(self.llvm_ir).add_current_process().export_symbol(
            "call_site")
        for name in global_addresses.keys():
            builder.export_symbol(name)
        self.__engine = builder.link(_lljit, f"{self.__id}_{self.__epoch}")
        self.__address = ctypes.c_char_p(self.__engine["call_site"])
        self.__other_sites = {name: ctypes.cast(self.__engine[name],
                                                ctypes.POINTER(ctypes.c_char_p)) for name in global_addresses.keys()}
        for name, address in global_addresses.items():
            ctypes.memmove(self.__other_sites[name], ctypes.addressof(address), ctypes.sizeof(ctypes.c_char_p))
        super().invalidate_address(self.__id, self.__address)
        self.__cfunc = self.__ctype(ctypes.cast(self.__address, ctypes.c_void_p).value)

    def invalidate_address(self, name: str, address: ctypes.c_char_p):
        # Do not call super as any listeners will have a direct reference to this address if they need it
        if name in self.__other_sites:
            ctypes.memmove(self.__other_sites[name], ctypes.addressof(address), ctypes.sizeof(ctypes.c_char_p))

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__handle.function_type()

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        value = flow.upsert_global_binding(self.__id, self.__type, self.__address)
        return flow.builder.call(flow.builder.load(value), args)


class CastHandle(Handle):
    """
    Create a handle that casts the parameters and return value of one handle to another.

    Each parameter is converted separately and all conversions are 1:1; that is, each parameter is converted using a
    handle that takes exactly one parameter and there is one handle for each parameter in the handle being cast in
    addition to one for the return type.

    It is not normally desirable to construct cast handles directly. The `cast` method on a handle will use the types to
    resolve the conversions automatically.
    """
    __handle: Handle
    __ret_handle: Handle
    __arg_handles: List[Handle]

    def __init__(self, handle: Handle, ret_handle: Handle, arg_handles: List[Handle]):
        super().__init__()
        handle.register(self)
        ret_handle.register(self)
        for arg_handle in arg_handles:
            arg_handle.register(self)
        self.__handle = handle
        self.__ret_handle = ret_handle
        self.__arg_handles = copy.copy(arg_handles)
        (ret_type, arg_types) = handle.function_type()
        assert (ret_type,) == ret_handle.function_type()[1], "Return type conversion doesn't match handle."
        assert len(arg_types) == len(arg_handles), "Argument conversions don't match handle."
        for index in range(0, len(arg_types)):
            handle_output = arg_handles[index].function_type()[0]
            assert arg_types[index] == handle_output,\
                f"Expected {arg_types[index]} at argument {index} but got {handle_output} from {arg_handles[index]}."

    def __str__(self) -> str:
        arg_str = ", ".join(str(h) for h in self.__arg_handles)
        return f"Cast ({self.__handle}) using {self.__ret_handle} ({arg_str})"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__ret_handle.function_type()[0], tuple(a.function_type()[1][0] for a in self.__arg_handles)

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        output_args = [flow.call(handle, (value,)) for (handle, value) in zip(self.__arg_handles, args)]
        return flow.call(self.__ret_handle, (flow.call(self.__handle, output_args),))


class IdentityHandle(Handle[ControlFlow]):
    """
    Create a handle which takes one input, of a particular type, and returns that value.
    """
    __ret: Type
    __arg: Type

    def __init__(self, ty: Type, arg_ty: Union[Type, None] = None):
        super().__init__()
        self.__ret = ty
        self.__arg = arg_ty or ty
        if arg_ty:
            assert arg_ty.machine_type() == ty.machine_type(), f"Cannot do no-op conversion from {arg_ty} to {ty}"

    def __str__(self) -> str:
        return f"Identity {self.__arg} → {self.__ret}"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__ret, (self.__arg,)

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        (arg, ) = args
        return arg


class PreprocessArgumentHandle(Handle):
    """
    Preprocesses a single argument of a handle using another handle.

    Unlike casting operations, this does not require a 1:1 match between handle types. Given a handle:
    `t f(t0, t1, t2, t3)` and `t2 g(s0, s1, s2)`, then preprocessing `f` using `g` at index 2, will produce a handle
    with the signature `t p(t0, t1, s0, s1, s2, t3)` which will behave as if `f(t0, t1, g(s0, s1, s2, s3), t3)`.
    """
    __handle: Handle
    __preprocessor: Handle
    __index: int

    def __init__(self, handle: Handle, index: int, preprocessor: Handle):
        super().__init__()
        handle.register(self)
        preprocessor.register(self)
        self.__handle = handle
        self.__preprocessor = preprocessor
        self.__index = index
        (_, args) = handle.function_type()
        handle_type = args[index]
        preprocessor_type = preprocessor.function_type()[0]
        assert handle_type == preprocessor_type,\
            f"Preprocessor produces {preprocessor_type} but handle expects {handle_type}"

    def function_type(self) -> (Type, Sequence[Type]):
        (handle_ret, handle_args) = self.__handle.function_type()
        (_, preprocess_args) = self.__preprocessor.function_type()
        args = []
        args.extend(handle_args[0:self.__index])
        args.extend(preprocess_args)
        args.extend(handle_args[self.__index + 1:])
        return handle_ret, args

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        preprocess_arg_len = len(self.__preprocessor.function_type()[1])
        handle_args = []
        handle_args.extend(args[0:self.__index])
        handle_args.append(
            flow.call(self.__preprocessor, args[self.__index:self.__index + preprocess_arg_len]))
        handle_args.extend(args[self.__index + preprocess_arg_len:])
        return flow.call(self.__handle, handle_args)

    def __str__(self) -> str:
        return f"Preprocess ({self.__handle}) at {self.__index} with ({self.__preprocessor})"


class IgnoreArgumentsHandle(Handle):
    """
    Creates a new handle that discards arguments before calling another handle.

    Depending on perspective, this drops or inserts arguments into a handle to allow it to discard unnecessary
    arguments. Given a handle `t f(t0, t1)`, dropping `e0` at index 1 would produce a handle with the signature:
    `t d(t0, e0, t1)`. Since arguments can be inserted at the end of the signature, the index can be the length of the
    original handle's argument list.
    """
    __handle: Handle
    __extra_args: Sequence[Type]
    __index: int

    def __init__(self, handle: Handle, index: int, *extra_args: Type):
        super().__init__()
        handle.register(self)
        self.__handle = handle
        self.__extra_args = extra_args
        self.__index = index
        (_, args) = handle.function_type()
        assert 0 <= index <= len(args), f"Index {index} for drop arguments is out of range [0, {len(args)}]"

    def function_type(self) -> (Type, Sequence[Type]):
        (handle_ret, handle_args) = self.__handle.function_type()
        args = []
        args.extend(handle_args[0:self.__index])
        args.extend(self.__extra_args)
        args.extend(handle_args[self.__index:len(handle_args)])
        return handle_ret, args

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        handle_args = []
        handle_args.extend(args[0:self.__index])
        handle_args.extend(args[self.__index + len(self.__extra_args):])
        return flow.call(self.__handle, handle_args)

    def __str__(self) -> str:
        return f"Ignore {self.__extra_args} at {self.__index} in {self.__handle}"


class DerefPointer(Handle):
    """
    Dereferences a pointer and returns the value

    The machine concept of a pointer is well-understood, but that does not make for a good type system. For instance,
    C's `FILE*` is a pointer, but it should never be de-referenced as the structure inside `FILE` should be an opaque
    implementation detail. The compromise that is made is that any `Type` can have an LLVM type that is a pointer, but
    the type system built on handles doesn't strictly _know_ that it is a pointer. That is, just because a type has an
    LLVM type that is a pointer, does not require that handles be allowed to dereference it. If a type wants to be
    dereferencable, it should extend `Deref` and provide the high-level type of its contents.

    This handle can operate in two ways: given a type that implement `Deref`, it can automatically figure out the
    corresponding return type. It can also be used in a coercive mode where both source and target types are provided
    and it only checks if the LLVM types are compatible.
    """
    __container_type: Type
    __target_type: Type

    def __init__(self, container_type: Type, target_ty: Union[Type, None] = None):
        super().__init__()
        self.__container_type = container_type
        if target_ty is None:
            if isinstance(container_type, Deref):
                self.__target_type = container_type.target()
            else:
                raise TypeError("Container type does not implement deref and no target type is not provided")
        else:
            self.__target_type = target_ty
            container_llvm_type = container_type.machine_type()
            assert isinstance(container_llvm_type, PointerType),\
                f"Container type must be a pointer but got {container_llvm_type}"
            assert container_llvm_type.pointee == target_ty.machine_type(),\
                f"Container points to {container_llvm_type.pointee} but target type is {target_ty.machine_type()}"

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        return flow.builder.load(args[0])

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__target_type, (self.__container_type,)

    def __str__(self) -> str:
        return f"Deref {self.__container_type} → {self.__target_type}"

