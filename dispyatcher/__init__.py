import copy
import ctypes
import weakref
from typing import Sequence, List, Dict, Union, Any

import llvmlite.binding
import llvmlite.ir
from llvmlite.ir import Type as LLType, IRBuilder, Value as IRValue


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
    def into_type(self, target):
        """
        Convert from self type into the target type provided.

        If a conversion is known, this function should return a handle with a signature that has a single parameter of
        its own type and a return type of the target provided. If no suitable conversion exists, the method should
        return `None`.

        This method operates in conjunction with `from_type` to allow either the source or destination type to provide
        a conversion, with the source type having priority.
        """
        pass

    def from_type(self, source):
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


class MachineType(Type):
    """
    A high-level type that operates on low-level machine types

    This would be the most direct mapping to C language types available.
    """
    __type: LLType

    def __init__(self, ty: LLType):
        """
        Construct a new instance of a machine type given the matching LLVM type
        :param ty:
        """
        self.__type = ty

    def __eq__(self, o: object) -> bool:
        if isinstance(o, MachineType):
            return self.__type == o.__type
        else:
            return False

    def __str__(self) -> str:
        return str(self.__type)

    def into_type(self, target):
        if isinstance(target, MachineType):
            self_is_int = isinstance(self.__type, llvmlite.ir.IntType)
            self_is_fp = is_llvm_floating_point(self.__type)
            other_is_int = isinstance(target.__type, llvmlite.ir.IntType)
            other_is_fp = is_llvm_floating_point(target.__type)
            if self_is_int and other_is_int:
                return IntegerResizeHandle(self, target)
            if self_is_fp and other_is_fp:
                return FloatResizeHandle(self, target)
            if self_is_int and other_is_fp:
                return IntegerToFloatHandle(self, target)
            if self_is_fp and other_is_int:
                return FloatToIntegerHandle(self, target)
            if isinstance(self.__type, llvmlite.ir.PointerType) and isinstance(target.__type, llvmlite.ir.PointerType):
                return BitCastHandle(self, target)
        return None

    def from_type(self, source):
        return None

    def machine_type(self) -> LLType:
        return self.__type


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


class Handle(InvalidationTarget):
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

    def cast(self, target_return_type: Type, *target_parameter_types: Type):
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
        pass

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        pass


_call_site_id = 0


def _next_call_site_id() -> str:
    global _call_site_id
    _call_site_id += 1
    return f"call_site_{_call_site_id}"


class CallSite(Handle):
    __ctype: ctypes.CFUNCTYPE
    __engine: Union[llvmlite.binding.ExecutionEngine, None]
    __handle: Handle
    __id: str
    __other_sites: Dict[str, ctypes.c_char_p]
    __type: llvmlite.ir.FunctionType
    __address: ctypes.c_char_p

    def __init__(self, handle):
        super().__init__()
        self.__id = _next_call_site_id()
        (ret_type, arg_types) = handle.function_type()
        self.__type = llvmlite.ir.FunctionType(ret_type.machine_type(), (a.machine_type() for a in arg_types))
        self.__ctype = ctypes.CFUNCTYPE(ret_type.ctypes_type(), *(a.ctypes_type() for a in arg_types))
        self.__handle = handle
        self.__engine = None
        self.__cfunc = None
        handle.register(self)
        self.invalidate()

    @property
    def address(self):
        return self.__address

    @property
    def cfunc(self):
        return self.__cfunc

    @property
    def handle(self) -> Handle:
        return self.__handle

    @handle.setter
    def handle(self, handle: Handle):
        assert handle.function_type() == self.__handle.function_type(), "Handle does not match call site signature."
        self.__handle.unregister(self)
        handle.register(self)
        self.__handle = handle
        self.invalidate()

    def invalidate(self):
        # Do not call super as any listeners should not have to update if using a call site as a method handle
        global_addresses = {}
        machine_triple = llvmlite.binding.Target.from_default_triple()
        module = llvmlite.ir.Module()
        module.triple = machine_triple.triple
        function = llvmlite.ir.Function(module, self.__type, "call_site")
        builder = IRBuilder(function.append_basic_block())
        builder.ret(self.__handle.generate_ir(builder, function.args, global_addresses))
        module.functions.append(function)
        if self.__engine:
            self.__engine.run_static_destructors()

        self.__engine = llvmlite.binding.create_mcjit_compiler(llvmlite.binding.parse_assembly(str(module)),
                                                               machine_triple.create_target_machine())
        self.__engine.finalize_object()
        self.__engine.run_static_constructors()
        self.__address = ctypes.c_char_p(self.__engine.get_function_address("call_site"))
        self.__other_sites = {name: ctypes.cast(self.__engine.get_global_value_address(name),
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

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        if self.__id in builder.module.globals:
            value = builder.module.globals[self.__id]
        else:
            value = llvmlite.ir.GlobalVariable(builder.module, self.__type.as_pointer(), self.__id)
            value.initializer = llvmlite.ir.Constant(self.__type.as_pointer(), llvmlite.ir.Undefined)
            global_addresses[self.__id] = self.__address

        return builder.call(builder.load(value), args)


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

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        output_args = [handle.generate_ir(builder, (value,), global_addresses) for (handle, value) in
                       zip(self.__arg_handles, args)]
        return self.__ret_handle.generate_ir(builder,
                                             (self.__handle.generate_ir(builder, output_args, global_addresses),),
                                             global_addresses)


class IdentityHandle(Handle):
    """
    Create a handle which takes one input, of a particular type, and returns that value.
    """
    __type: Type

    def __init__(self, ty: Type):
        super().__init__()
        self.__type = ty

    def __str__(self) -> str:
        return f"Identity {self.__type}"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__type, (self.__type,)

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        (arg, ) = args
        return arg


class IntegerResizeHandle(Handle):
    """
    Convert an integer to the correct size through signed extension or truncation, as appropriate.

    The machine types for both of the types provided must be `llvm.ir.IntType`. This handle assumes the type conversion
    is acceptable for the types provided; it only cares about the machine precision.
    """
    __source: Type
    __target: Type

    def __init__(self, source: Type, target: Type):
        super().__init__()
        assert isinstance(source.machine_type(), llvmlite.ir.IntType), "Source type must have an integer machine type"
        assert isinstance(target.machine_type(), llvmlite.ir.IntType), "target type must have an integer machine type"
        self.__source = source
        self.__target = target

    def __str__(self) -> str:
        return f"Convert {self.__source} → {self.__target}"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__target, (self.__source,)

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        (arg, ) = args
        if self.__source.machine_type().width > self.__target.machine_type().width:
            return builder.trunc(arg, self.__target.machine_type())
        if self.__source.machine_type().width < self.__target.machine_type().width:
            return builder.sext(arg, self.__target.machine_type())
        return arg


class FloatResizeHandle(Handle):
    """
    Convert a floating-point to the correct size through extension or truncation, as appropriate.

    The machine types for both of the types provided must be `llvm.ir.HalfType`, `llvm.ir.FloatType`, or
    `llvm.ir.DoubleType`. This handle assumes the type conversion is acceptable for the types provided; it only cares
    about the machine precision.
    """
    __source: Type
    __target: Type

    def __init__(self, source: Type, target: Type):
        super().__init__()
        assert is_llvm_floating_point(source.machine_type()), "Source type must have a floating-point machine type"
        assert is_llvm_floating_point(target.machine_type()), "Target type must have a floating-point machine type"
        self.__source = source
        self.__target = target

    def __str__(self) -> str:
        return f"Convert f{self.__source} → f{self.__target}"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__target, (self.__source,)

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        def width(ty: llvmlite.ir.Type) -> int:
            if isinstance(ty, llvmlite.ir.HalfType):
                return 16
            if isinstance(ty, llvmlite.ir.FloatType):
                return 32
            if isinstance(ty, llvmlite.ir.DoubleType):
                return 64
            raise "Invalid floating point type"
        (arg, ) = args
        source_width = width(self.__source.machine_type())
        target_machine_type = self.__target.machine_type()
        target_width = width(target_machine_type)
        if source_width > target_width:
            return builder.fptrunc(arg, self.__target.machine_type())
        if source_width < target_width:
            return builder.fpext(arg, self.__target.machine_type())
        return arg


class IntegerToFloatHandle(Handle):
    """
    Converts an integer type to a floating point type

    The source type must have a machine type of `llvm.ir.IntType` and the target must have a machine type of
    `llvm.ir.HalfType`, `llvm.ir.FloatType`, or `llvm.ir.DoubleType`. This handle assumes the type conversion is
    acceptable for the types provided; it only cares about the machine precision.
    """
    __source: Type
    __target: Type

    def __init__(self, source: Type, target: Type):
        super().__init__()
        assert isinstance(source.machine_type(), llvmlite.ir.IntType), "Source is not an integer type"
        assert is_llvm_floating_point(target.machine_type()), "Target is not a floating point type"
        self.__source = source
        self.__target = target

    def __str__(self) -> str:
        return f"Convert {self.__source} → f{self.__target}"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__target, (self.__source,)

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        (arg, ) = args
        return builder.sitofp(arg, self.__target.machine_type())


class FloatToIntegerHandle(Handle):
    """
    Converts an integer type to a floating point type

    The source type must have a machine type of `llvm.ir.IntType` and the target must have a machine type of
    `llvm.ir.HalfType`, `llvm.ir.FloatType`, or `llvm.ir.DoubleType`. This handle assumes the type conversion is
    acceptable for the types provided; it only cares about the machine precision.
    """
    __source: Type
    __target: Type

    def __init__(self, source: Type, target: Type):
        super().__init__()
        assert is_llvm_floating_point(source.machine_type()), "Source is not a floating point type"
        assert isinstance(target.machine_type(), llvmlite.ir.IntType), "Target is not an integer type"
        self.__source = source
        self.__target = target

    def __str__(self) -> str:
        return f"Convert f{self.__source} → {self.__target}"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__target, (self.__source,)

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        (arg, ) = args
        return builder.fptosi(arg, self.__target.machine_type())


class BitCastHandle(Handle):
    """
    Performs an LLVM bit-cast to reinterpret the memory of one type as another.

    This handle assumes the type conversion is acceptable for the types provided; it does not validate that conversion
    will yield valid results.
    """
    __source: Type
    __target: Type

    def __init__(self, source: Type, target: Type):
        super().__init__()
        self.__source = source
        self.__target = target

    def __str__(self) -> str:
        return f"BitCast {self.__source} → {self.__target}"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__target, (self.__target,)

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        (arg, ) = args
        return builder.bitcast(arg, self.__target.machine_type())


class NoOpHandle(Handle):
    """
    Performs type conversion where the underlying types have the same machine type and need to conversion.

    This handle assumes the type conversion is acceptable for the types provided; it does not validate that conversion
    will yield valid results.
    """
    __source: Type
    __target: Type

    def __init__(self, source: Type, target: Type):
        super().__init__()
        assert source.machine_type() == target.machine_type(), "Machine types are not compatible."
        self.__source = source
        self.__target = target

    def __str__(self) -> str:
        return f"NoOp {self.__source} → {self.__target}"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__target, (self.__target,)

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
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
        args.extend(handle_args[self.__index:len(handle_args)])
        return handle_ret, args

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        preprocess_arg_len = len(self.__preprocessor.function_type()[1])
        handle_args = []
        handle_args.extend(args[0:self.__index])
        handle_args.append(
            self.__preprocessor.generate_ir(builder, args[self.__index:self.__index + preprocess_arg_len],
                                            global_addresses))
        handle_args.extend(args[self.__index + preprocess_arg_len: len(args)])
        return self.__handle.generate_ir(builder, handle_args, global_addresses)

    def __str__(self) -> str:
        return f"Preprocess ({self.__handle}) at {self.__index} with ({self.__preprocessor}"


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

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        handle_args = []
        handle_args.extend(args[0:self.__index])
        handle_args.extend(args[self.__index + len(self.__extra_args):])
        return self.__handle.generate_ir(builder, handle_args, global_addresses)

    def __str__(self) -> str:
        return f"Ignore {self.__extra_args} at {self.__index} in {self.__handle}"


class SimpleConstantHandle(Handle):
    """
    A handle that returns a constant value

    A handle with no arguments that returns a constant value. The constant value must be one that can be encoded in LLVM
    IR, which includes integers, floating points, arrays of the former, and strings. The exact LLVM IR produce will
    depend on the machine type.
    """
    __value: Any
    __type: Type

    def __init__(self, ty, value):
        super().__init__()
        self.__type = ty
        self.__value = value

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__type, ()

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        return llvmlite.ir.Constant(self.__type.machine_type(), self.__value)

    def __str__(self) -> str:
        return f"Constant {self.__type} ({self.__value})"


llvmlite.binding.initialize()
llvmlite.binding.initialize_native_target()
llvmlite.binding.initialize_native_asmprinter()
