import ctypes
import enum
import graphviz
import llvmlite.binding
import llvmlite.ir
import os
import weakref
from llvmlite.ir import Type as LLType, IRBuilder, Value as IRValue, PointerType
from typing import Sequence, List, Dict, TypeVar, Generic, Tuple, Set, Optional, Union, Callable, Any

_config = os.environ.get("DISPYATCHER_DEBUG", "").split(",")

dump_ir = "dump-ir" in _config
"""
If set to true, or ``dump-ir`` is set in the environment variable ``DISPYATCHER_DEBUG``, all callsites will be printed
to standard output on compilation
"""

dump_gv = "dump-gv" in _config
"""
If set to true, or ``dump-gv`` is set in the environment variable ``DISPYATCHER_DEBUG``, all callsites will generate a
GraphViz graph on compilation and it will be written to a file. If the call site is recompiled, the next iteratio will
be written to a different file.
"""


def llvm_type_to_ctype(ty: LLType):
    """
    Find a ``ctype`` representation from an LLVM type

    This function finds the appropriate ctype for an LLVM type, if one exists. The ctype representation may lose
    information about the LLVM type, especially where structures are concerned. Some LLVM types do not have equivalent
    ctype representations including half-precision floats (aka ``float16``), vectors, metadata types, and label types.

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
    Checks if the LLVM type provided is a floating point type.

    :param ty: the LLVM type to check
    :return: true if an instance of ``llvm.ir.HalfType``, ``llvm.ir.FloatType``, or ``llvm.ir.DoubleType``
    """
    return (isinstance(ty, llvmlite.ir.HalfType) or isinstance(ty, llvmlite.ir.FloatType) or
            isinstance(ty, llvmlite.ir.DoubleType))


class Type:
    """
    The representation of a parameter or return type that a handle can use
    """
    def into_type(self, target) -> Optional["Handle"]:
        """
        Convert from self type into the target type provided.

        If a conversion is known, this function should return a target with a signature that has a single parameter of
        its own type and a return type of the target provided. If no suitable conversion exists, the method should
        return ``None``.

        This method operates in conjunction with ``from_type`` to allow either the source or destination type to provide
        a conversion, with the source type having priority.
        """
        return None

    def from_type(self, source) -> Optional["Handle"]:
        """
        Convert from the provided type into the self type.

        If a conversion is known, this function should return a target with a signature that has a single parameter of
        the source type provided and a return its own type. If no suitable conversion exists, the method should return
        ``None``.

        This method operates in conjunction with ``into_type`` to allow either the source or destination type to provide
        a conversion, with the source type having priority.
        """
        return None

    def clone(self, flow: "F", value: IRValue) -> IRValue:
        """
        Creates a copy of a value of this type and returns the copy.

        "Copy" is a type-specific idea. Since some handles will require an "owned" value and making a "copy" is meant to
        satisfy that requirement. For some types, a copy might just be the value; *e.g.*, a copy of an integer is just
        the original value. Similarly, pointers to constant values (*e.g.*, function pointers, string literals), can
        also be copied by using the original value. For Python objects or other reference counted objects, the copy
        operation can simply adjust the reference count and return the same value, while tracking objects, *e.g.*, C++'s
        ``std::shared_ptr<>``, may return a different value. It's also worth noting that the underlying type does not
        determine the behaviour of cloning. For instance, it would be possible to create a type for a file descriptor,
        which is represented as an integer, but still requires a copy to create an independently closable descriptor.

        If the value cannot be copied, this should throw an exception.

        :param flow: the control flow in which to generate the copy code
        :param value: the value to copy
        :return: the copied value
        """
        raise ValueError(f"Type {self} can't be copied")

    def clone_is_self_contained(self) -> bool:
        """
        Indicates if a copy of a value of this type may be a self-contained copy or a copy that still references the
        lifetimes of objects that made it.

        For instance, creating a copy of an iterator may still reference the original collection and be bound by its
        lifetime; while creating a copy of that collection may have no lifetime requirements. It's possible to have a
        copy operation that will preserve some lifetimes and not others, but this is not supported.

        :return: whether the value is self-contained (true) or holds on to the existing lifetimes (false)
        """
        raise NotImplementedError()

    def ctypes_type(self):
        """
        Provide the ctype representation of this type

        If no ``ctype`` representation exists, it should raise a type error. The default method for this method will
        return the ``ctype`` representation of the machine/LLVM type. It is provided in the case where a better ctype
        representation is available.
        """
        return llvm_type_to_ctype(self.machine_type())

    def drop(self, flow: "FlowState", value: IRValue) -> None:
        """
        Destroys a value

        Calls the destructor on a value, if necessary. It is assumed that a destructor will be called for any owned
        values that are not returned.

        Many values will be trivially destroyable (*e.g.*, numeric values, function pointers), in which case this method
        should simply generate no codes.

        It is also possible to have a value which should never be destroyed (*i.e.*, a monad), in which case, this
        method can throw an exception.

        :param flow: the control flow in which to generate the destruction code
        :param value: the value to destroy
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def as_pointer(self) -> "Type":
        """
        Creates a new type that is a simple pointer to this type.

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
        raise NotImplementedError()


class Pointer(Deref):
    """
    A type for a simple pointer

    This, at the machine level, looks like a C++ ``&`` reference or a Rust reference. Unlike a C ``*`` pointer, no
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

    def clone(self, builder: IRBuilder, value: IRValue) -> IRValue:
        return value

    def clone_is_self_contained(self) -> bool:
        return False

    def drop(self, flow: "FlowState", value: IRValue) -> None:
        self.__inner.drop(flow, flow.builder.load(value))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Pointer):
            return o.__inner == self.__inner
        return False

    def __str__(self) -> str:
        return f"Pointer({self.__inner})"


ArgumentManagement = enum.Enum('ArgumentManagement', [
    'BORROW_TRANSIENT',
    'BORROW_CAPTURE',
    'BORROW_CAPTURE_PARENTS',
    'TRANSFER_TRANSIENT',
    'TRANSFER_CAPTURE_PARENTS'])
"""
This determines how memory and lifetime management will work for an argument. There are two parts: the memory management
and the lifetime model.

The first part determines memory management:

* ``BORROW``: the caller is responsible for freeing this value when it is not in use
* ``TRANSFER``: the callee is responsible for freeing this value

Normally arguments are assumed to be borrowed for the lifetime of the function, and they can be destroyed after the
function is called. However, some functions take ownership of the argument and the caller should not destroy the
argument.

For some types that have trivial copy functions, the distinction between owning or not owning the argument might be
meaningless though it is still made. If you are familiar with Rust, these are types that would implement the `Copy`
trait. It is preferred to transfer in these cases, but it should not affect the generated code as the inserted memory
management will be no-ops.

The second part determines the lifetime management. For any given handle the result may be tied to the input arguments.
That is to say that it might be valid only if the input value has not been destroyed. This might be because the value
references an internal part of the other structure (*e.g.*, it is a pointer to an element of a collection, or a chunk of
an arena or an iterator over a collection).

A handle can make the distinction between capturing the lifetime of an argument vs capturing the same lifetimes as that
argument.

* ``TRANSIENT``: the function may read the value, but the output does not depend on this value
* ``CAPTURE``: the function will produce output that depends on the lifetime of this argument (*i.e.*, the result must
    be freed before this argument is freed)
* ``CAPTURE_PARENTS``: the function will produce output that depends on the lifetimes of the lifetimes already captured
    by this value, but not the value itself (consider a situation like an iterator; this indicates the function depends
    on the collection, but not the iterator itself)

For example, suppose you take an iterator over a collection as an argument and capture a value from the iterator. The
resulting data does not require the iterator to continue existing upon returning. However, it does require the
collection the iterator was borne from continue to exist. This is the distinction made been a ``CAPTURE`` and
``CAPTURE_PARENTS``.

Note that ``TRANSFER_CAPTURE`` is an impossible condition as it means the callee is going to free a value, but the
output is dependent on it.
"""


ReturnManagement = enum.Enum('ReturnManagement', ['BORROW', 'TRANSFER'])
"""
This determines how memory management will work for a return value:

* ``BORROW``: the callee is responsible for freeing this value when it is not in use
* ``TRANSFER``: the caller is responsible for freeing this value

For instance, if the value is an element of a collection, it shouldn't invoke a destructor directly; the value can be
discarded without calling an explicit destructor. This is not true for an iterator that would require freeing the memory
associated with the iterator even if the collection will live on.
"""


class FlowState:
    """
    A flow state captures the current function being built to implement a call site.

    It exists as a separate class from a control flow because it is possible to change control flows when implementing a
    call site and a single call site might need to have independent flow states if it has branching paths.
    """
    __arguments: Set[int]
    __builder: IRBuilder
    __global_addresses: Dict[str, ctypes.c_size_t]
    __library_dependencies: Set[str]
    __dependencies: Dict[int, Tuple[Set[int], IRValue, Optional[Type]]]
    __next_lifetime: int

    def __init__(self,
                 builder: IRBuilder,
                 dependencies: Union[int, Dict[int, Tuple[Set[int], IRValue, Optional[Type]]]],
                 global_addresses: Dict[str, ctypes.c_size_t],
                 library_dependencies: Set[str]):
        self.__arguments = set(range(dependencies)) if isinstance(dependencies, int)else set(dependencies.keys())
        self.__builder = builder
        self.__global_addresses = global_addresses
        self.__library_dependencies = library_dependencies
        if isinstance(dependencies, int):
            self.__dependencies = {arg: (set(), builder.function.args[arg], None) for arg in range(0, dependencies)}
            self.__next_lifetime = dependencies
        else:
            self.__dependencies = dependencies
            self.__next_lifetime = max(self.__dependencies.keys()) + 1 if dependencies else 0

    @property
    def builder(self) -> IRBuilder:
        """
        Access the LLVM IR builder for this flow control.
        """
        return self.__builder

    def add_library_dependency(self, name: str) -> None:
        """
        Adds an OrcJIT dependency to the callsite.

        :param name: the name of the library
        """
        self.__library_dependencies.add(name)

    def check_read(self, lifetime: int) -> None:
        """
        Assert tha that a lifetime is still valid.

        :param lifetime: the identifier of the lifetime
        """
        assert lifetime in self.__dependencies, "Trying to read lifetime that has been transferred or dropped"

    def create_lifetime(self, value: IRValue, parents: Sequence[Tuple[int, bool]], ty: Optional[Type]) -> int:
        """
        Tracks a new lifetime.

        :param value: the LLVM value that needs to be tracked
        :param parents: the other lifetimes to connect to this lifetime; each is an argument index and a Boolean value
            indicating whether to include this value (``False``) or the lifetimes of its ancestors (``True``)
        :param ty: the type of the value, if it needs to be freed explicitly. If does not, None should be provided and
            the drop operation will be a no-op (though dependant values may still get explicit drop operations)
        :return: an integer which is the state-specific lifetime identifier
        """
        lifetime = self.__next_lifetime
        self.__next_lifetime += 1
        lifetimes = set()
        for parent, inner in parents:
            if inner:
                dependencies = self.__dependencies[parent]
                assert dependencies, f"Failed to create lifetime as {parent} is an argument"
                lifetimes.update(dependencies[0])
            else:
                lifetimes.add(parent)
        self.__dependencies[lifetime] = (lifetimes, value, ty)
        return lifetime

    def drop(self, lifetime: int) -> None:
        """
        Explicitly drop a lifetime and any dependant values.

        This will run all appropriate destructors.

        :param lifetime: the state-specific lifetime identifier
        """
        (_, value, ty) = self.__dependencies[lifetime]
        self.transfer(lifetime)
        if ty is not None:
            self.builder.comment(f"Destroying owned {value} with lifetime {lifetime}")
            ty.drop(self, value)
        else:
            self.builder.comment(f"Discarding borrowed {value} with lifetime {lifetime}")

    def drop_all(self) -> None:
        """
        Drops all values. This is only useful when trying to handle exceptional control flow
        """
        active = list(sorted(self.__dependencies.keys(), reverse=True))
        for a in active:
            self.drop(a)

    def finish(self, return_lifetime: Union[None, int, "TemporaryValue"]) -> None:
        """
        Terminates a flow state, possibly with a return value.

        This will trigger cleanup of all temporary values in the block. If a return lifetime is provided, that value
        will be spared from the cleanup and it will be checked that it does not depend on any temporary values that are
        being cleaned.

        :param return_lifetime: an optional return lifetime to track
        """
        spared_lifetimes = set(self.__arguments)
        if isinstance(return_lifetime, TemporaryValue):
            spared_lifetimes.add(return_lifetime.lifetime)
            return_lifetime.pluck()
            self.builder.comment(f"Finishing with lifetime {return_lifetime}")
        elif isinstance(return_lifetime, int):
            spared_lifetimes.add(return_lifetime)
            self.pluck(return_lifetime)
            self.builder.comment(f"Finishing with lifetime {return_lifetime}")
        else:
            self.builder.comment("Finishing with no return lifetime")
        active = list(sorted((lt for lt in self.__dependencies.keys() if lt not in spared_lifetimes), reverse=True))
        for a in active:
            self.drop(a)

    def fork(self) -> "FlowState":
        """
        Create a partial-copy of this flow state.

        This partial copy will share global state with the main flow state, including the LLVM IR generator, library
        dependency tracking, and global addresses. It will have an independent set of lifetimes. This allows a handle to
        create a child control flow for implementing branching paths. Each fork must be finished. Lifetime identifiers
        **cannot** be shared across flow states and are not guaranteed to be globally unique.

        :return: the new flow state that will share the same global addresses and builder as this one
        """
        return FlowState(self.__builder, {**self.__dependencies}, self.__global_addresses, self.__library_dependencies)

    def pluck(self, lifetime: int) -> None:
        """
        Plucks a value out of the current flow state. This is an advanced version of transfer that ensure that the value
        not only has no dependant values, but it also only depends on the initial lifetimes of a flow state. For a call
        site, the initial lifetimes are that of the arguments. For a forked value, any values that were live at the time
        of forking as allowed as parent lifetimes of the plucked value.

        :param lifetime: the lifetime identifier to pluck
        """
        (parents, _, _) = self.__dependencies[lifetime]
        for parent in parents:
            assert parent in self.__arguments, f"Trying to prune {lifetime}, but it depends on temporary {parent}."
        self.transfer(lifetime)

    def transfer(self, lifetime: int) -> None:
        """
        Transfers to another controller a lifetime by dropping any dependant values.

        :param lifetime: the lifetime to transfer
        """
        del self.__dependencies[lifetime]
        deps = list(sorted((dead_child for dead_child, (parents, _, _) in self.__dependencies.items()
                            if lifetime in parents), reverse=True))
        if deps:
            self.builder.comment(f"Eliminating {lifetime} requires eliminating dependant lifetimes {deps}")
            for dep in deps:
                self.drop(dep)

    def upsert_global_binding(self, name: str, ty: LLType, address: ctypes.c_size_t) -> IRValue:
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
            value = llvmlite.ir.GlobalVariable(self.__builder.module, ty, name)
            value.initializer = llvmlite.ir.Constant(ty, llvmlite.ir.Undefined)
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
        if name in self.builder.module.globals:
            return self.builder.module.globals[name]
        else:
            value = llvmlite.ir.Function(self.builder.module,
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
        if name in self.builder.module.globals:
            return self.builder.module.globals[name]
        else:
            value = llvmlite.ir.GlobalVariable(self.builder.module, ty, name)
            return value


class TemporaryValue:
    """
    The result of a call value that can be passed to another call or returned
    """
    __state: FlowState
    __lifetime: int
    __live: bool
    __value: IRValue

    def __init__(self, state: FlowState, value: IRValue, lifetime: int):
        self.__state = state
        self.__value = value
        self.__lifetime = lifetime
        self.__live = True

    def __str__(self) -> str:
        return f"TemporaryValue[{self.__lifetime}]({self.__value})"

    @property
    def ir_value(self) -> IRValue:
        """
        The LLVM value that is the result of the call
        """
        return self.__value

    @property
    def lifetime(self) -> int:
        return self.__lifetime

    def check_read(self) -> None:
        """
        Ensure that this lifetime is still alive

        Throws an exception if the value has been transferred or dropped
        """
        assert self.__live, "Cannot check value"
        self.__state.check_read(self.__lifetime)

    def drop(self) -> None:
        """
        Discards this value and generates the associated cleanup
        """
        assert self.__live, "Cannot drop dead value"
        self.__state.drop(self.__lifetime)
        self.__live = False

    def pluck(self) -> IRValue:
        """
        Drops any dependant values and makes sure that the result has no dependencies outside of arguments

        :return: the LLVM value
        """
        assert self.__live, "Cannot transfer dead value"
        self.__state.pluck(self.__lifetime)
        self.__live = False
        return self.__value

    def transfer(self) -> IRValue:
        """
        Drops any dependant values and transfers control of this value

        Note that this is what you should call if returning the result of a call.

        :return: the LLVM value
        """
        assert self.__live, "Cannot transfer dead value"
        self.__state.transfer(self.__lifetime)
        self.__live = False
        return self.__value


class ControlFlow:
    """
    A non-linear flow control mechanism

    This super type can be used for a control flow that does not support alternate control flow. There is a separate
    control flow that allow using the CPython exception mechanism for non-linear flow. The callsite must have one
    top-level flow control, but the handles within don't have to match exactly. It is possible to have handles that can
    adapt from one flow control to another (*e.g.*, check ``errno`` and turn it into a Python exception) or if flow
    controls are logical subsets of one another (_e.g._, a handle using this flow control, which is infallible/linear,
    can be used inside a callsite that has takes another flow control); said another way, you can always use a handle
    that doesn't throw in a callsite that handles a throw.
    """
    __state: FlowState
    # The ar lifetimes are:
    # - the calling handle
    # - the argument lifetimes: True is value is owned and not transferred yet, False if transferred, otherwise a set of
    #                           lifetime identifiers
    # - a list of clean up callbacks
    __arg_lifetimes: List[Tuple["Handle", List[Union[bool, Set[int]]], List[Callable[[], None]]]]

    def __init__(self, state: FlowState):
        self.__state = state
        self.__arg_lifetimes = []

    @property
    def builder(self) -> IRBuilder:
        """
        Access the LLVM IR builder for this flow control.
        """
        return self.__state.builder

    def add_library_dependency(self, name: str) -> None:
        """
        Adds an OrcJIT dependency to the callsite.

        :param name: the name of the library
        """
        self.__state.add_library_dependency(name)

    def call(self, handle: "Handle", args: Sequence[Union[TemporaryValue, Tuple[IRValue, Sequence[int]]]]) ->\
            TemporaryValue:
        """
        Calls another handle.

        This is the correct way to invoke another handle to ensure memory management is done correctly.

        :param handle: the handle to call
        :param args: the arguments to pass to that handle; each argument can be the output of an previously called
            handle or a raw value connected to the caller's handle arguments (*i.e.*, the index of the handle's
            arguments indicates how the lifetime of the result of this function is connected to the caller's own input)
        :return: the return value from that handle

        """
        (callee_return_type, callee_return_management) = handle.handle_return()
        callee_argument_info = handle.handle_arguments()
        assert len(args) == len(callee_argument_info), "Wrong number of arguments in call"
        if self.__arg_lifetimes:
            (caller, caller_arg_lifetimes, _) = self.__arg_lifetimes[-1]
            caller_argument_info = caller.handle_arguments()
        else:
            # This is the base case trigger by the call site, so we fake everything to the same as the handle
            caller = "callsite"
            caller_arg_lifetimes = [{i} for i in range(len(callee_argument_info))]
            caller_argument_info = callee_argument_info
        # Indicates which caller arguments are used by the callee
        argument_lifetime_mapping = {}
        # Indicates which callee arguments are transfers or what associated lifetime identifiers are used
        callee_argument_lifetimes = []
        values = []
        self.builder.comment(f"Preparing arguments for {handle}...")
        lifetimes_to_check = set()
        for index, (arg, (arg_type, arg_management)) in enumerate(zip(args, callee_argument_info)):
            if isinstance(arg, TemporaryValue):
                values.append(arg.ir_value)
                if arg_management in (ArgumentManagement.TRANSFER_TRANSIENT,
                                      ArgumentManagement.TRANSFER_CAPTURE_PARENTS):
                    self.builder.comment(f"Transferring argument {index} for {handle}...")
                    arg.transfer()
                    callee_argument_lifetimes.append(True)
                else:
                    lifetimes_to_check.add(arg.lifetime)
                    callee_argument_lifetimes.append({arg.lifetime})
            else:
                (value, caller_args) = arg
                values.append(value)
                if arg_management in (ArgumentManagement.TRANSFER_TRANSIENT,
                                      ArgumentManagement.TRANSFER_CAPTURE_PARENTS):
                    for caller_arg in caller_args:
                        assert caller_arg_lifetimes[caller_arg] is True,\
                            (f"Handle {caller} transfers argument {caller_arg} to handle {handle} for argument {index} "
                             " but doesn't own it (any longer)")
                        self.builder.comment(f"Transferring {caller_arg} argument to {index} for {handle}...")
                        caller_arg_lifetimes[caller_arg] = False
                    callee_argument_lifetimes.append(True)
                elif arg_management == ArgumentManagement.BORROW_TRANSIENT:
                    callee_argument_lifetimes.append(set())
                    for caller_arg in caller_args:
                        lifetimes_for_arg = caller_arg_lifetimes[caller_arg]
                        if isinstance(lifetimes_for_arg, set):
                            lifetimes_to_check.update(lifetimes_for_arg)
                        else:
                            assert lifetimes_for_arg is True, \
                                (f"Handle {caller} transfers argument {caller_arg} to handle {handle} for argument"
                                 f" {index}  but doesn't own it (any longer)")
                else:
                    combined_caller_lifetimes = set()
                    for caller_arg in caller_args:
                        lifetimes_for_arg = caller_arg_lifetimes[caller_arg]
                        if isinstance(lifetimes_for_arg, set):
                            combined_caller_lifetimes.update(lifetimes_for_arg)
                            (_, caller_arg_management) = caller_argument_info[caller_arg]
                            capture_parents = caller_arg_management in (ArgumentManagement.TRANSFER_CAPTURE_PARENTS,
                                                                        ArgumentManagement.BORROW_CAPTURE_PARENTS)
                            for lifetime in lifetimes_for_arg:
                                lifetimes_to_check.add(lifetime)
                                if lifetime not in argument_lifetime_mapping:
                                    argument_lifetime_mapping[lifetime] = capture_parents
                                else:
                                    argument_lifetime_mapping[lifetime] &= capture_parents
                        else:
                            assert lifetimes_for_arg is True, \
                                (f"Handle {caller} transfers argument {caller_arg} to handle {handle} for argument"
                                 f" {index}  but doesn't own it (any longer)")
                    callee_argument_lifetimes.append(combined_caller_lifetimes)
        # We delay checking all the lifetimes until after the transfers are done. This avoids the situation where a
        # handle borrows and then transfers the same value
        for lifetime in lifetimes_to_check:
            self.__state.check_read(lifetime)
        self.__arg_lifetimes.append((handle, callee_argument_lifetimes, []))
        self.builder.comment(f"Calling {handle}...")
        value = handle.generate_handle_ir(self, values)
        _, _, cleanups = self.__arg_lifetimes.pop()
        if isinstance(value, IRValue):
            lifetime = self.__state.create_lifetime(
                value,
                [(lifetime, capture_inner) for lifetime, capture_inner in argument_lifetime_mapping.items()],
                callee_return_type if callee_return_management == ReturnManagement.TRANSFER else None)
            self.builder.comment(f"Created {lifetime} for {value}, as result of {handle}")
            value = TemporaryValue(self.__state, value, lifetime)

        if cleanups:
            self.builder.comment(f"Cleaning up after call to {handle}")
            for cleanup in cleanups:
                cleanup()

        return value

    def call_and_pluck(self, handle: "Handle", args: Sequence[Union[TemporaryValue, Tuple[IRValue, Sequence[int]]]]) ->\
            IRValue:
        branched_flow = self._create_branch(self.__state.fork())
        branched_flow.__arg_lifetimes.append(self.__arg_lifetimes[-1])
        result = branched_flow.call(handle, args)
        branched_flow.__state.finish(result)
        return result.ir_value

    def _create_branch(self, state) -> "F":
        """
        Creates a new control flow that is a separate of the existing flow for a branched execution pattern.

        :param state: the control flow state to use
        :return: the new control flow state
        """
        return ControlFlow(state)

    def drop_arg(self, index: int) -> None:
        """
        Triggers generation of a drop for an argument. This is only safe if the argument is transferred.

        :param index: the position of the argument as seen by the current handle
        """
        (caller, caller_arg_lifetimes, _) = self.__arg_lifetimes[-1]
        if caller_arg_lifetimes[index]:
            self.builder.comment(f"Manually dropping argument {index} for {caller}")
            for lifetime in caller_arg_lifetimes[index]:
                self.__state.drop(lifetime)

    def fork_and_die(self) -> None:
        """
        Creates a new control flow state and then drops all known values.

        This is only useful for handling exception program flow.
        """
        self.__state.fork().drop_all()

    def unwind_cleanup(self, cleanup: Callable[[], None]) -> None:
        """
        Register a callback that will be executed when the current handle frame is exited.

        There is no guarantee when this will execute relative to any lifetime management.

        :param cleanup: the callback to execute
        """
        self.__arg_lifetimes[-1][2].append(cleanup)

    def upsert_global_binding(self, name: str, ty: LLType, address: ctypes.c_size_t) -> IRValue:
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
        return self.__state.upsert_global_binding(name, ty, address)

    def use_native_function(self, name: str, ret_ty: LLType, args: Sequence[LLType], var_arg: bool = False) -> IRValue:
        """
        Create a binding for a function exported form the current binary.

        :param name: the name to use
        :param ret_ty: the LLVM type for the return of the function
        :param args: the LLVM type for the return of the function
        :param var_arg: whether the function takes variadic arguments
        :return: an LLVM global constant for this binding
        """
        return self.__state.use_native_function(name, ret_ty, args, var_arg)

    def use_native_global(self, name: str, ty: LLType) -> IRValue:
        """
        Create a binding for a symbol exported form the current binary.

        :param name: the name to use
        :param ty: the LLVM type for the symbol
        :return: an LLVM global constant for this binding
        """
        return self.__state.use_native_global(name, ty)


_COLOUR_PALETTE = [
    "#eaaecf",
    "#afc987",
    "#74aff3",
    "#eedea5",
    "#53c6ef",
    "#ecaa9a",
    "#52c6cf",
    "#ddbc98",
    "#7dd5e6",
    "#bcb67d",
    "#c4b7ea",
    "#99cc9a",
    "#9dbbe6",
    "#d2f0be",
    "#98cdf1",
    "#c8cd9c",
    "#97ece1",
    "#91bc9f",
    "#a2ddbd",
    "#85c8bb"]


class DiagramState:
    """
    Holds a DOT/Graphviz diagram of a handle and all of its children
    """
    __graph: graphviz.Digraph
    __colour_table: Dict[str, int]
    __call_stack: List[Tuple["Handle", str, str]]
    __id_generator: int

    def __init__(self, handle: "Handle"):
        """
        Create a new diagram for a handle
        """
        self.__graph = graphviz.Digraph('Handle', strict=False)
        self.__colour_table = {}
        self.__call_stack = []
        self.__id_generator = 0

        self.__graph.attr(compound="true")

        arg_labels = "|".join(f"<arg{idx}> {idx}: {graphviz.escape(str(ty))}"
                              for idx, (ty, _) in enumerate(handle.handle_arguments()))
        label = f"{{ Input | {{{arg_labels}}}}}"
        self.__graph.node("input", graphviz.nohtml(label), shape="record")
        result = self.call(handle, [f"input:arg{idx}" for idx in range(len(handle.handle_arguments()))])
        self.__graph.node("output", str(handle.handle_return()[0]), shape="doublecircle")
        self.__graph.edge(result, "output")

    def call(self, handle: "Handle", args: Sequence[str]) -> str:
        """
        Adds a new call to a graph directing to a provided handle.

        :param handle: the handle to call
        :param args: the graph identifiers for the arguments to that handle
        :return: a graph identifier for the result
        """
        name = type(handle).__qualname__
        if name not in self.__colour_table:
            self.__colour_table[name] = len(self.__colour_table)
        node_colour = _COLOUR_PALETTE[self.__colour_table[name] % len(_COLOUR_PALETTE)]
        self.__call_stack.append((handle, name, node_colour))
        result = handle.generate_handle_diagram(self, args)
        self.__call_stack.pop()
        return result

    def dispatch(self, cases: Sequence[Tuple[str, "Handle"]], args: Sequence[str]) -> str:
        node_id = f"cluster_dispatch{self.__id_generator}"
        node_id_input = f"dispatch_input{self.__id_generator}"
        node_id_output = f"dispatch_output{self.__id_generator}"
        self.__id_generator += 1
        g = self.__graph
        (dispatch_handle, name, node_colour) = self.__call_stack[-1]
        with self.__graph.subgraph(name=node_id) as s:
            s.attr(label=name, color=node_colour, style="fill")
            dispatch_args = dispatch_handle.handle_arguments()
            arg_labels = "|".join(f"<arg{idx}> {idx}: {graphviz.escape(str(ty))}"
                                  for idx, (ty, _) in enumerate(dispatch_args))
            label = f"{{ {{{arg_labels}}} | Dispatch Input}}"
            s.node(node_id_input, graphviz.nohtml(label), shape="record")

            for idx, arg in enumerate(args):
                g.edge(arg, f"{node_id_input}:arg{idx}")

            outputs = []
            for idx, (label, handle) in enumerate(cases):
                case_name = f"{node_id}_case{idx}"
                case_input = f"{case_name}_input"
                with s.subgraph(name=case_name) as case_graph:
                    case_graph.attr(label=label, color="#dfdfdf", style="fill")
                    arg_labels = "|".join(f"<arg{idx}> {idx}: {graphviz.escape(str(ty))}"
                                          for idx, (ty, _) in enumerate(handle.handle_arguments()))
                    label = f"{{ Case Input | {{{arg_labels}}}}}"
                    case_graph.node(case_input, graphviz.nohtml(label), shape="record")
                    s.edge(node_id_input, case_input, style="dashed")
                    self.__graph = case_graph
                    outputs.append(self.call(handle, [f"{case_input}:arg{idx}"
                                                      for idx in range(len(handle.handle_arguments()))]))

            s.node(node_id_output, str(dispatch_handle.handle_return()[0]), shape="doublecircle")
            for output in outputs:
                s.edge(output, node_id_output)

        self.__graph = g
        return node_id_output

    def simple_handle(self, args: Sequence[str]) -> str:
        """
        Renders a graph node as a block with slots for arguments and the name of the handle.

        A colour will be automatically picked. This is the default and no intervention is required to use it.

        :param args: the graph identifiers for the arguments to that handle
        :return: a graph identifier for the result
        """
        node_id = f"handle{self.__id_generator}"
        self.__id_generator += 1
        (handle, name, node_colour) = self.__call_stack[-1]
        handle_description = handle.handle_description()
        (ret_ty, ret_mgmt) = handle.handle_return()
        lines = [
            "{" + "|".join(f"<arg{idx}> {idx}: {graphviz.escape(str(ty))}[{mgmt.name}]"
                           for idx, (ty, mgmt) in enumerate(handle.handle_arguments())) + "}",
            name
        ]
        if handle_description:
            lines.append(graphviz.escape(handle_description))

        lines.append(f"<ret>{ret_ty}[{ret_mgmt.name}]")
        self.__graph.node(node_id,
                          graphviz.nohtml("{" + " | ".join(lines) + "}"),
                          shape="record",
                          style="filled",
                          fillcolor=node_colour)
        for idx, arg in enumerate(args):
            self.__graph.edge(arg, f"{node_id}:arg{idx}")
        return f"{node_id}:ret"

    def transform(self, arg: str, note: str) -> str:
        """
        Creates an edge with a label for simple operations.

        Some 1:1 handles do trivial operations (*e.g.*, a null check) and creating a block would clutter the diagram.
        This provides an alternative representation for those handles.

        :param arg: the graph identifier of the single input
        :param note: the label to apply to the operation
        :return: the graph identifier for the result
        """
        node_id = f"note{self.__id_generator}"
        self.__id_generator += 1
        self.__graph.node(node_id, note, shape="none")
        self.__graph.edge(arg, node_id)
        return node_id

    def wrap(self, handle: "Handle", args: Sequence[str]) -> str:
        """
        Create a block with a handle inside it.

        Some handles wrap calls to other handles to perform stateful operations (*e.g.*, with GIL). This is the
        preferred representation for those handles.

        :param handle: the inner handle being wrapped
        :param args: the arguments to the inner handle
        :return: the graph identifier for the result
        """
        node_id = f"wrap{self.__id_generator}"
        self.__id_generator += 1
        g = self.__graph
        (_, name, node_colour) = self.__call_stack[-1]
        with self.__graph.subgraph(name=node_id, label=name, color=node_colour) as s:
            self.__graph = s
            result = self.call(handle, args)

        self.__graph = g
        return result

    def render(self, filename, output_format: str = 'png') -> None:
        """
        Renders the diagram as an image.

        :param filename: the output filename to store the image; this should not include a suffix because GraphViz
        :param output_format: the file format supported by GraphViz
        """
        self.__graph.render(filename=filename, format=output_format, cleanup=True)

    def render_and_load(self, output_format: str = 'png') -> bytes:
        """
        Renders the diagram as an image and provides it as a byte array.

        :param output_format: the file format supported by GraphViz
        :return: the image as bytes
        """
        return self.__graph.pipe(format=output_format)

    def save(self, filename) -> None:
        """
        Stores the GraphViz description of this handle in a file.

        :param filename: the filename
        """
        self.__graph.save(filename=filename)

    def saves(self) -> str:
        """
        Stores the GraphViz description of this handle in a string

        :return: the GraphViz data
        """
        return self.__graph.source


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

    def invalidate_address(self, name: str, address: ctypes.c_size_t) -> None:
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

    def invalidate_address(self, name: str, address: ctypes.c_size_t) -> None:
        """
        Trigger invalidation of the address of a call site and propagate that to its listeners.

        :param name: the internal name of the call site being used as a handle
        :param address:  the new address of the function to use
        """
        for listener in self.__listeners:
            listener.invalidate_address(name, address)

    def register(self, listener: InvalidationListener) -> None:
        """
        Add a listener to receive invalidation updates for this object.

        :param listener: the listener to notify when changes happen
        """
        self.__listeners.add(listener)

    def unregister(self, listener: InvalidationListener) -> None:
        """
        Remove a listener to stop receiving invalidation updates.

        :param listener:  the listener to remove
        """
        self.__listeners.remove(listener)


F = TypeVar("F", bound=ControlFlow)


class Handle(InvalidationTarget, Generic[F]):
    """
    A handle is a small function-like behaviour that can be composed and converted to machine code

    Handles, based on the Java Virtual Machine's ``MethodHandle``, are a way to dynamically change a kind of special
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

    def __add__(self, other) -> "Handle":
        if isinstance(other, Handle):
            return PreprocessArgument(other, 0, self)
        elif callable(other):
            return PreprocessArgument(other(self.handle_return()[0]), 0, self)
        elif isinstance(other, tuple) and callable(other[0]):
            return PreprocessArgument(other[0](self.handle_return()[0], *other[1:]), 0, self)
        else:
            raise TypeError(f"Unsupported operand type for +: {type(other)}")

    def __floordiv__(self, other) -> "Handle":
        if isinstance(other, Type):
            (ret_type, ret_management) = self.handle_return()
            if ret_type == other:
                return self
            processor = ret_type.into_type(other) or other.from_type(ret_type)
            if not processor:
                raise TypeError(f"No conversion from {ret_type} to {other} in cast of {self}.")
            return PreprocessArgument(processor, 0, self)
        else:
            raise TypeError(f"Unsupported operand type for //: {type(other)}")

    def __lshift__(self, other) -> "Handle":
        if isinstance(other, Handle):
            return PreprocessArgument(self, 0, other)
        elif isinstance(other, tuple):
            if len(other) == 2 and isinstance(other[0], int) and isinstance(other[1], Handle):
                return PreprocessArgument(self, other[0], other[1])

            result = self
            for index, handle in reversed(list(enumerate(other))):
                if handle is None:
                    continue
                elif isinstance(handle, Handle):
                    result = PreprocessArgument(result, index, handle)
                else:
                    raise TypeError(f"Unsupported type for at {index} for <<: {type(handle)}")
            return result
        else:
            raise TypeError(f"Unsupported operand type for <<: {type(other)}")

    def __truediv__(self, other) -> "Handle":
        output = self
        args = self.handle_arguments()
        assert len(args) == len(other), f"Handle takes {len(args)}, but {len(other)} provided by cast."
        for idx, (src_type, (tgt_type, _)) in enumerate(zip(other, args)):
            assert isinstance(tgt_type, Type), f"Argument type {tgt_type} is not a type"
            if tgt_type == src_type:
                continue
            processor = src_type.into_type(tgt_type) or tgt_type.from_type(src_type)
            if not processor:
                raise TypeError(f"No conversion from {src_type} to {tgt_type} in cast of {self}.")
            output = PreprocessArgument(output, idx, processor)
        return output

    def __matmul__(self, other):
        if callable(other):
            return other(self)
        elif isinstance(other, tuple) and callable(other[0]):
            return other[0](self, *other[1:])
        else:
            raise TypeError(f"Unsupported operand type for @: {type(other)}")

    # noinspection PyMethodMayBeStatic
    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        """
        Draws a diagram of how this node operates.

        Most subclasses don't need to override this as it will produce a sensible little block. Handles that manipulate
        flow control should to create a better diagram.

        :param diagram: the diagram being rendered to
        :param args: the graph identifiers for the arguments to this handle
        :return: the graph identifier that is the output of this handle
        """
        return diagram.simple_handle(args)

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        """
        Convert the handle into LLVM machine code.

        :param flow: the flow control builder
        :param args: the arguments to the handle
        :return: the value that is the output of the handle
        """
        pass

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        """
        Gets the argument information of the handle as a sequence of argument type and management pairs.

        :return: a sequence of tuples with the type and management for each argument
        """
        pass

    def handle_description(self) -> Optional[str]:
        """
        Generates a description of the handle for display in graphs

        :return: the description
        """
        return None

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        """
        Gets the return information of a handle as a pair of the type and memory management semantics.

        Note that argument semantics determine the lifetime of the output, so that information is not provided here.

        :return: a pair of the return type and the memory management of the return value
        """
        pass

    def show_diagram(self) -> DiagramState:
        """
        Renders this handle as a DOT graph for debugging.

        :return: the diagram
        """
        return DiagramState(self)


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

    def create_flow(self, state: FlowState,
                    return_type: Type,
                    return_management: ReturnManagement,
                    arg_types: Sequence[Tuple[Type, ArgumentManagement]]) ->\
            F:
        """
        Constructs a new control flow for a single compilation pass.

        :param state: the fixed information for the control flow
        :param return_type: the return type of the callsite
        :param return_management: the return management of the callsite
        :param arg_types: the argument types of the callsite
        :return: the control flow created
        """
        return ControlFlow(state)

    def bridge_function(self,
                        ret: Tuple[Type, ReturnManagement],
                        arguments: Sequence[Tuple[Type, ArgumentManagement]],
                        address: int) -> Callable[..., Any]:
        """
        Creates an appropriate Python function for the call site to invoke when called, if possible.

        This can create any synthetic parameters (*e.g.*, callbacks for asynchronous flow) or register the calling
        convention with respect to the Python GIL. Ultimately, this will be invoked when the callite is used as a
        callable in Python and can do whatever is appropriate for this flow. If the signature means that the handle
        cannot be used safely from Python, this function can throw an exception.

        :param ret: the return information of the callsite
        :param arguments: the argument information of the callsite
        :param address: the address of the compiled callsite
        :return: the function type
        """
        for idx, (arg_type, arg_management) in enumerate(arguments):
            if arg_management in (ArgumentManagement.TRANSFER_TRANSIENT, ArgumentManagement.TRANSFER_CAPTURE_PARENTS):
                message = f"Argument {idx} of type {arg_type} cannot be safely transferred from external caller"

                def bad_transfer(*args, **kwargs):
                    raise ValueError(message)
                return bad_transfer
        return ctypes.CFUNCTYPE(ret[0].ctypes_type(), *(a.ctypes_type() for a, _ in arguments))(address)


llvmlite.binding.initialize()
llvmlite.binding.initialize_native_target()
llvmlite.binding.initialize_native_asmprinter()

lljit_instance = llvmlite.binding.create_lljit_compiler()


class CallSite(Handle):
    """
    A callsite is a wrapper around a handle that allows that handle to be compiled and called from Python.

    Callsites can also be used as handles in other callsites and independently updated.
    """
    __address: ctypes.c_size_t
    __engine: Optional[llvmlite.binding.ResourceTracker]
    __epoch: int
    __flow_type: ControlFlowType
    __func: Callable[..., Any]
    __handle: Handle[F]
    __id: str
    __other_sites: Dict[str, ctypes.POINTER(ctypes.c_size_t)]
    __type: llvmlite.ir.FunctionType
    llvm_ir: str

    def __init__(self, handle: Handle[F], flow_type: ControlFlowType = ControlFlowType()):
        """
        Create a new callsite that wraps an existing handle.

        :param handle: the handle to place in the callsite; it can be updated later, but a handle must be supplied
        :param flow_type: the control flow to be used for the callsite; the handle must be compatible with this flow
        """
        super().__init__()

        def not_initialized(*_args, **_kwargs):
            raise ValueError("Callsite is not yet compiled")

        self.__id = _next_call_site_id()
        self.__epoch = 0
        (ret_type, _) = handle.handle_return()
        self.__type = llvmlite.ir.FunctionType(ret_type.machine_type(), (a.machine_type()
                                                                         for a, _ in handle.handle_arguments()))
        self.__flow_type = flow_type
        self.__handle = handle
        self.__engine = None
        self.__func = not_initialized
        self.llvm_ir = "; Not yet compiled"
        handle.register(self)
        self.invalidate()

    @property
    def address(self) -> ctypes.c_size_t:
        """
        The address of the compiled version of the callsite (*i.e.*, the function pointer that references inside the
        callsite.

        Note that this is not guaranteed to be a stable address. If the callsite is regenerated, this address may
        change. To track the address, use the invalidation subscription mechanism.

        :return: the address to the compiled contents of the callsite
        """
        return self.__address

    @property
    def handle(self) -> Handle[F]:
        """
        The current handle inside the call site.

        :return: the handle
        """
        return self.__handle

    @handle.setter
    def handle(self, handle: Handle[F]):
        assert (handle.handle_return() == self.__handle.handle_return() and
                tuple(handle.handle_arguments()) == tuple(self.__handle.handle_arguments())),\
            "Handle does not match call site signature."
        self.__handle.unregister(self)
        handle.register(self)
        self.__handle = handle
        self.invalidate()

    def __call__(self, *args, **kwargs):
        return self.__func(*args, **kwargs)

    def invalidate(self):
        self.__epoch += 1
        unique_id = f"{self.__id}_{self.__epoch}"

        if dump_gv:
            DiagramState(self.__handle).save(unique_id + ".gv")

        # Do not call super as any listeners should not have to update if using a call site as a method handle
        machine_triple = llvmlite.binding.Target.from_default_triple()
        module = llvmlite.ir.Module()
        module.triple = machine_triple.triple
        function = llvmlite.ir.Function(module, self.__type, "call_site")
        builder = IRBuilder(function.append_basic_block())
        global_addresses = {}
        library_dependencies = set()
        state = FlowState(builder, len(self.__handle.handle_arguments()), global_addresses, library_dependencies)
        flow = self.__flow_type.create_flow(state,
                                            self.__handle.handle_return()[0],
                                            self.__handle.handle_return()[1],
                                            self.__handle.handle_arguments())
        result = flow.call(self.__handle, [(a, (i,)) for i, a in enumerate(function.args)])
        state.finish(result)
        builder.ret(result.ir_value)
        module.functions.append(function)

        self.llvm_ir = str(module)
        if dump_ir:
            print(self.llvm_ir)

        builder = llvmlite.binding.JITLibraryBuilder().add_ir(self.llvm_ir).add_current_process().export_symbol(
            "call_site")
        for name in global_addresses.keys():
            builder.export_symbol(name)
        for library_dependency in library_dependencies:
            builder.add_jit_library(library_dependency)
        self.__engine = builder.link(lljit_instance, unique_id)
        self.__address = ctypes.c_size_t(self.__engine["call_site"])
        self.__other_sites = {name: ctypes.cast(self.__engine[name],
                                                ctypes.POINTER(ctypes.c_size_t)) for name in global_addresses.keys()}
        for name, address in global_addresses.items():
            ctypes.memmove(self.__other_sites[name], ctypes.addressof(address), ctypes.sizeof(ctypes.c_size_t))
        super().invalidate_address(self.__id, self.__address)
        self.__func = self.__flow_type.bridge_function(self.handle_return(),
                                                       self.handle_arguments(),
                                                       self.__address.value)

    def invalidate_address(self, name: str, address: ctypes.c_size_t):
        # Do not call super as any listeners will have a direct reference to this address if they need it
        if name in self.__other_sites:
            ctypes.memmove(self.__other_sites[name], ctypes.addressof(address), ctypes.sizeof(ctypes.c_size_t))

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return self.__handle.handle_arguments()

    def handle_description(self) -> Optional[str]:
        return self.__id

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__handle.handle_return()

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        value = flow.upsert_global_binding(self.__id, self.__type.as_pointer(), self.__address)
        return flow.builder.call(flow.builder.load(value), args)

    def show_diagram(self) -> DiagramState:
        """
        Renders this handle as a DOT graph for debugging.

        :return: the diagram
        """
        return DiagramState(self.__handle)


class BaseTransferUnaryHandle(Handle[F], Generic[F]):
    """
    This handle serves as a base for unary operators that can operate in borrowing and owning modes.

    There are several operations that take a single input, do some trivial operation that does not change the lifetimes
    of that input, and returns the modified output. These operations are often agnostic to whether the input is owned or
    borrowed. At the most trivial, the identity handle that simply returns its input falls in this category. This handle
    provides the necessary implementation for lifetime and transfer handling for these operations.
    """
    __ret: Type
    __arg: Type
    __management: ReturnManagement

    def __init__(self, ty: Type, arg_ty: Type, transfer: ReturnManagement):
        super().__init__()
        self.__ret = ty
        self.__arg = arg_ty
        self.__management = transfer

    @property
    def return_management(self) -> ReturnManagement:
        return self.__management

    def __str__(self) -> str:
        return f"{self._name()}[{self.__management.name}]({self.__arg}) → {self.__ret}"

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        arg_management = (ArgumentManagement.BORROW_CAPTURE_PARENTS
                          if self.__management == ReturnManagement.BORROW else
                          ArgumentManagement.TRANSFER_CAPTURE_PARENTS)
        return (self.__arg, arg_management),

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__ret, self.__management

    def _name(self) -> str:
        raise NotImplementedError()


class Clone(Handle[ControlFlow]):
    """
    Create a handle which takes one input and returns a copy/cloned value.
    """
    __type: Type

    def __init__(self, ty: Type):
        super().__init__()
        self.__type = ty

    def __str__(self) -> str:
        return f"Clone[{self.__type}]"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        (arg,) = args
        return diagram.transform(arg, "Clone")

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[IRValue, TemporaryValue]:
        (arg, ) = args
        return self.__type.clone(flow, arg)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        if self.__type.clone_is_self_contained():
            arg_management = ArgumentManagement.BORROW_TRANSIENT
        else:
            arg_management = ArgumentManagement.BORROW_CAPTURE_PARENTS
        return (self.__type, arg_management),

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__type, ReturnManagement.TRANSFER


class DerefPointer(Handle):
    """
    Dereferences a pointer and returns the value

    The machine concept of a pointer is well-understood, but that does not make for a good type system. For instance,
    C's ``FILE*`` is a pointer, but it should never be de-referenced as the structure inside ``FILE`` should be an
    opaque implementation detail. The compromise that is made is that any ``Type`` can have an LLVM type that is a
    pointer, but the type system built on handles doesn't strictly _know_ that it is a pointer. That is, just because a
    type has an LLVM type that is a pointer, does not require that handles be allowed to dereference it. If a type wants
    to be dereferencable, it should extend ``Deref`` and provide the high-level type of its contents.

    This handle can operate in two ways: given a type that implement ``Deref``, it can automatically figure out the
    corresponding return type. It can also be used in a coercive mode where both source and target types are provided
    and it only checks if the LLVM types are compatible.
    """
    __container_type: Type
    __target_type: Type

    def __init__(self, container_type: Type, target_ty: Optional[Type] = None):
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

    def __str__(self) -> str:
        return f"Deref({self.__container_type}) → {self.__target_type}"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[IRValue, TemporaryValue]:
        return flow.builder.load(args[0])

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return (self.__container_type, ArgumentManagement.BORROW_CAPTURE),

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__target_type, ReturnManagement.BORROW


class Identity(BaseTransferUnaryHandle[ControlFlow]):
    """
    Create a handle which takes one input, of a particular type, and returns that value.
    """

    def __init__(self, ty: Type, arg_ty: Optional[Type] = None, transfer: ReturnManagement = ReturnManagement.BORROW):
        super().__init__(ty, arg_ty or ty, transfer)
        if arg_ty:
            assert arg_ty.machine_type() == ty.machine_type(), f"Cannot do no-op conversion from {arg_ty} to {ty}"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        (arg,) = args
        return arg

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[IRValue, TemporaryValue]:
        (arg, ) = args
        return arg

    def _name(self) -> str:
        return "Identity"


IgnoreCapture = enum.Enum('IgnoreCapture', ['TRANSIENT', 'CAPTURE', 'CAPTURE_PARENTS'])


class IgnoreArguments(Handle):
    """
    Creates a new handle that discards arguments before calling another handle.

    Depending on perspective, this drops or inserts arguments into a handle to allow it to discard unnecessary
    arguments. Given a handle ``t f(t0, t1)``, dropping ``e0`` at index 1 would produce a handle with the signature:
    ``t d(t0, e0, t1)``. Since arguments can be inserted at the end of the signature, the index can be the length of the
    original handle's argument list.

    All arguments are borrowed. If you need to drop them, apply a take ownership handle after.
    """
    __handle: Handle
    __extra_args: Sequence[Type]
    __index: int
    __capture: IgnoreCapture

    def __init__(self, handle: Handle, index: int, *extra_args: Type, capture: IgnoreCapture = IgnoreCapture.TRANSIENT):
        super().__init__()
        handle.register(self)
        self.__handle = handle
        self.__extra_args = extra_args
        self.__index = index
        self.__capture = capture
        num_args = len(handle.handle_arguments())
        assert 0 <= index <= num_args, f"Index {index} for drop arguments is out of range [0, {num_args}]"

    def __str__(self) -> str:
        args = ', '.join(str(a) for a in self.__extra_args)
        return f"Ignore[{self.__capture.name}]({args}) at {self.__index} of {self.__handle}"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        handle_args = []
        handle_args.extend(a for a in args[0:self.__index])
        handle_args.extend(a for a in args[self.__index + len(self.__extra_args):])
        return diagram.call(self.__handle, handle_args)

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[IRValue, TemporaryValue]:
        handle_args = []
        handle_args.extend((a, (index,)) for index, a in enumerate(args[0:self.__index]))
        handle_args.extend((a, (self.__index + index,))
                           for index, a in enumerate(args[self.__index + len(self.__extra_args):]))
        return flow.call(self.__handle, handle_args)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        if self.__capture == IgnoreCapture.TRANSIENT:
            management = ArgumentManagement.BORROW_TRANSIENT
        elif self.__capture == IgnoreCapture.CAPTURE:
            management = ArgumentManagement.BORROW_CAPTURE
        else:
            management = ArgumentManagement.BORROW_CAPTURE_PARENTS
        original = self.__handle.handle_arguments()
        output = []
        output.extend(original[0:self.__index])
        output.extend((t, management) for t in self.__extra_args)
        output.extend(original[self.__index:])
        return output

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__handle.handle_return()


class PreprocessArgument(Handle):
    """
    Preprocesses a single argument of a handle using another handle.

    Unlike casting operations, this does not require a 1:1 match between handle types. Given a handle:
    ``t f(t0, t1, t2, t3)`` and ``t2 g(s0, s1, s2)``, then preprocessing ``f`` using ``g`` at index 2, will produce a
    handle with the signature ``t p(t0, t1, s0, s1, s2, t3)`` which will behave as if
    ``f(t0, t1, g(s0, s1, s2, s3), t3)``.
    """
    __handle: Handle
    __preprocessor: Handle
    __index: int
    __needs_copy: bool

    def __init__(self, handle: Handle, index: int, preprocessor: Handle):
        super().__init__()
        handle.register(self)
        preprocessor.register(self)
        self.__handle = handle
        self.__preprocessor = preprocessor
        self.__index = index
        (arg_type, arg_management) = handle.handle_arguments()[index]
        (preprocessor_type, preprocessor_management) = preprocessor.handle_return()
        assert arg_type == preprocessor_type,\
            f"Preprocessor [{preprocessor}] produces {preprocessor_type} but handle [{handle}] expects {arg_type}"
        self.__needs_copy = (preprocessor_management == ReturnManagement.BORROW and arg_management in
                             (ArgumentManagement.TRANSFER_TRANSIENT, ArgumentManagement.TRANSFER_CAPTURE_PARENTS))

    def __str__(self) -> str:
        return f"Preprocess ({self.__handle}) at {self.__index} with ({self.__preprocessor})"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        preprocess_arg_len = len(self.__preprocessor.handle_arguments())
        preprocessed_result = diagram.call(self.__preprocessor,
                                           [a for a in args[self.__index:self.__index + preprocess_arg_len]])
        if self.__needs_copy:
            preprocessed_result = diagram.transform(preprocessed_result, "Clone")
        handle_args = []
        handle_args.extend(a for a in args[0:self.__index])
        handle_args.append(preprocessed_result)
        handle_args.extend(a for a in args[self.__index + preprocess_arg_len:])
        return diagram.call(self.__handle, handle_args)

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[IRValue, TemporaryValue]:
        preprocess_arg_len = len(self.__preprocessor.handle_arguments())
        preprocessed_result = flow.call(self.__preprocessor, [(args[idx], (idx,)) for idx in
                                                              range(self.__index, self.__index + preprocess_arg_len)])
        if self.__needs_copy:
            preprocessed_result = self.__preprocessor.handle_return()[0].clone(flow, preprocessed_result.ir_value)
        handle_args = []
        handle_args.extend((args[idx], (idx,)) for idx in range(self.__index))
        handle_args.append(preprocessed_result)
        handle_args.extend((args[idx], (idx,)) for idx in range(self.__index + preprocess_arg_len, len(args)))
        return flow.call(self.__handle, handle_args)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        handle_args = self.__handle.handle_arguments()
        preprocess_args = self.__preprocessor.handle_arguments()
        args = []
        args.extend(handle_args[0:self.__index])
        args.extend(preprocess_args)
        args.extend(handle_args[self.__index + 1:])
        return args

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__handle.handle_return()


class TakeOwnership(Handle):
    """
    Create a handle which wraps another handle and takes ownership of arguments and frees them immediately after the
    inner handle finishes.
    """
    __handle: Handle
    __owned_args: Sequence[int]

    def __init__(self, handle: Handle, *owned_args: int):
        super().__init__()
        self.__handle = handle
        self.__owned_args = owned_args
        assert not owned_args or max(owned_args) < len(handle.handle_arguments()),\
            f"Arguments {owned_args} out of range for {handle}"
        for idx, (arg_type, arg_management) in enumerate(handle.handle_arguments()):
            assert arg_management != ArgumentManagement.BORROW_CAPTURE or idx not in owned_args, \
                f"Handle {handle} captures argument {idx}; taking ownership would be a lifetime violation"

    def __str__(self) -> str:
        return f"TakeOwnership[{', '.join(str(a) for a in self.__owned_args)}] of {self.__handle}"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        return diagram.wrap(self.__handle, args)

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[IRValue, TemporaryValue]:
        result = flow.call(self.__handle, [(a, (i,)) for i, a in enumerate(args)])
        for owned_argument in self.__owned_args:
            flow.drop_arg(owned_argument)
        return result

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        sub = {
            ArgumentManagement.BORROW_TRANSIENT: ArgumentManagement.TRANSFER_TRANSIENT,
            ArgumentManagement.BORROW_CAPTURE_PARENTS: ArgumentManagement.TRANSFER_CAPTURE_PARENTS,
        }
        return [(t, sub.get(a, a) if idx in self.__owned_args else a)
                for idx, (t, a) in enumerate(self.__handle.handle_arguments())]

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__handle.handle_return()
