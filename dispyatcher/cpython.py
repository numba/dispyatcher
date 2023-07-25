import ctypes
from typing import Any, Callable, List, NamedTuple, Sequence, Set, Tuple, Union, Optional

import llvmlite.ir
from llvmlite.ir import IRBuilder, Value as IRValue, Type as LLType

import dispyatcher.general
import llvmlite.ir.types
from dispyatcher import ArgumentManagement, BaseTransferUnaryHandle, ControlFlow, DerefPointer, DiagramState, F,\
    FlowState, Handle, Identity, Type, ReturnManagement, TemporaryValue
from dispyatcher.accessors import GetElementPointer
from dispyatcher.general import BaseIndirectFunction, CurrentProcessFunction, MachineType, UncheckedArray
from dispyatcher.permute import implode_args
from dispyatcher.repacking import Repacker, RepackingDispatcher, RepackingState

INT_RESULT_TYPE = llvmlite.ir.types.IntType(ctypes.sizeof(ctypes.c_int) * 8)
SIZE_T_TYPE = llvmlite.ir.types.IntType(ctypes.sizeof(ctypes.c_size_t) * 8)
CHAR_ARRAY_TYPE = UncheckedArray(MachineType(llvmlite.ir.IntType(8)))
LONG_TYPE = MachineType(llvmlite.ir.IntType(ctypes.sizeof(ctypes.c_long) * 8))


def _ptr_to_obj(value: Any) -> ctypes.c_size_t:
    import platform
    assert platform.python_implementation() == "CPython", \
        "This feature only works on CPython because there's no universal way to get the address of an object."
    # We rely on id to return the pointer to an object, which is a problem.
    return ctypes.c_size_t(id(value))


class PythonControlFlow(dispyatcher.ControlFlow):
    """
    A control flow that knows how to generate Python exceptions
    """
    __return_type: LLType

    def __init__(self, state: FlowState, return_type: LLType):
        super().__init__(state)
        self.__return_type = return_type

    def check_return_code(self, result: IRValue, ok_value: int, error: str, message: str) -> None:
        """
        Many CPython functions return an integer for a true/false/exception case. This is a helper to build the flow
        logic for these functions.

        :param result: the result returned by the function
        :param ok_value: the return value that signals the happy path
        :param error: the exception name when the function is on the unhappy non-exception path
        :param message: the message to raise on the unhappy non-exception path
        """
        fail_block = self.builder.append_basic_block("result_fail")
        ok_block = self.builder.append_basic_block("result_ok")
        exception_block = self.builder.append_basic_block("result_exception")
        switch = self.builder.switch(result, fail_block)
        switch.add_case(llvmlite.ir.Constant(INT_RESULT_TYPE, ok_value), ok_block)
        switch.add_case(llvmlite.ir.Constant(INT_RESULT_TYPE, -1), exception_block)

        self.builder.position_at_start(fail_block)
        self.throw_exception(error, message)
        self.builder.position_at_start(exception_block)
        self.unwind()
        self.builder.position_at_start(ok_block)

    def throw_exception(self, error: str, message: str) -> None:
        """
        Throw an exception and stop the current control flow.

        :param error: the name of the exception (only the built-in exceptions such as ``ValueError``)
        :param message: the message to put into the exception
        """
        set_string = self.use_native_function("PyErr_SetString",
                                              llvmlite.ir.VoidType(),
                                              (PY_OBJECT_TYPE.machine_type(), llvmlite.ir.IntType(8).as_pointer()))
        exception_name = "PyExc_" + error
        exception = self.use_native_global(exception_name, PY_OBJECT_TYPE.machine_type())
        message_bytes = bytearray(message.encode('utf-8'))
        message_bytes.append(0)
        message_type = llvmlite.ir.ArrayType(llvmlite.ir.IntType(8), len(message_bytes))
        message_value = llvmlite.ir.GlobalVariable(self.builder.module,
                                                   message_type,
                                                   self.builder.module.get_unique_name("exception"))
        message_value.initializer = message_type(message_bytes)
        self.builder.call(set_string,
                          (self.builder.load(exception),
                           self.builder.bitcast(message_value, llvmlite.ir.IntType(8).as_pointer())))
        self.unwind()

    def unwind(self) -> None:
        """
        Unwind the current cleanup and return a junk value.

        This is helpful if the callee has set the Python exception state.
        """
        self.fork_and_die()
        self.builder.ret(self.__return_type(None))


class PythonControlFlowType(dispyatcher.ControlFlowType):
    """
    The control flow that allows CPython exceptions to be thrown
    """

    def create_flow(self,
                    state: FlowState,
                    return_type: Type,
                    return_management: ReturnManagement,
                    arg_types: Sequence[Tuple[Type, ArgumentManagement]]) -> F:
        assert not isinstance(return_type, PyObjectType) or return_management == ReturnManagement.TRANSFER,\
            "A Python control flow must return an owned object."

        return PythonControlFlow(state, return_type.machine_type())

    def bridge_function(self, ret: Tuple[Type, ReturnManagement],
                        arguments: Sequence[Tuple[Type, ArgumentManagement]],
                        address: int) -> Callable[..., Any]:
        return ctypes.PYFUNCTYPE(ret[0].ctypes_type(), *(a.ctypes_type() for a, _ in arguments))(address)


class PyObjectType(Type):
    """
    The type of a Python object, using the ``PyObject*`` at the machine level, that also knows a Python type and can
    insert appropriate type checks during handle conversions.
    """
    __type: type

    def __init__(self, ty: type):
        assert isinstance(ty, type), f"Expected type but got {ty}"
        self.__type = ty

    def __eq__(self, o: object) -> bool:
        if isinstance(o, PyObjectType):
            return self.__type == o.__type
        else:
            return False

    def __str__(self) -> str:
        return f"PyObject({self.__type.__name__})"

    @property
    def python_type(self) -> type:
        return self.__type

    def clone(self, flow: PythonControlFlow, value: IRValue) -> IRValue:
        inc_block = flow.builder.append_basic_block("incref_nonnull")
        end_block = flow.builder.append_basic_block("incref_end")
        is_some = flow.builder.icmp_unsigned('!=', value, self.machine_type()(None))
        flow.builder.cbranch(is_some, inc_block, end_block)

        flow.builder.position_at_start(inc_block)

        inc_ref = flow.use_native_function("Py_IncRef", llvmlite.ir.VoidType(), (PY_OBJECT_TYPE.machine_type(),))
        flow.builder.call(inc_ref, (value,))
        flow.builder.branch(end_block)
        flow.builder.position_at_start(end_block)
        return value

    def clone_is_self_contained(self) -> bool:
        return True

    def ctypes_type(self):
        return ctypes.py_object

    def drop(self, flow: FlowState, value: IRValue) -> None:
        dec = flow.builder.append_basic_block("decref_nonnull")
        end_block = flow.builder.append_basic_block("decref_end")
        is_some = flow.builder.icmp_unsigned('!=', value, self.machine_type()(None))
        flow.builder.cbranch(is_some, dec, end_block)

        flow.builder.position_at_start(dec)

        dec_ref = flow.use_native_function("Py_DecRef", llvmlite.ir.VoidType(), (PY_OBJECT_TYPE.machine_type(),))
        flow.builder.call(dec_ref, (value,))
        flow.builder.branch(end_block)
        flow.builder.position_at_start(end_block)

    def into_type(self, target: Type) -> Union[Handle, None]:
        if isinstance(target, PyObjectType):
            if issubclass(self.__type, target.__type):
                return Identity(target, self)
            else:
                return CheckedCast(self.__type, target.__type)

    def machine_type(self) -> LLType:
        return llvmlite.ir.PointerType(llvmlite.ir.IntType(8))


PY_OBJECT_TYPE = PyObjectType(object)
""" The type for an unknown Python object (in C, `PyObject*`) """

PY_DICT_TYPE = PyObjectType(dict)
""" The type for a Python dictionary """


class AlwaysThrow(Handle[PythonControlFlow]):
    """
    A handle that always throws an exception.
    """
    error: str
    exception: str
    __ty: Type
    __transfer: ReturnManagement

    def __init__(self,
                 ty: Type,
                 transfer: ReturnManagement = ReturnManagement.TRANSFER,
                 exception: str = "ValueError",
                 error: str = "Value cannot be null"):
        """
        Create a new handle which checks if the argument is not null and raises an exception if it is null.

        :param ty: the type that will be returned; although this handle will never return, it still needs a type
        :param transfer: whether the return value is transferred; although this will never return, it still needs a
            management
        :param exception: the exception type to throw
        :param error: the error message to raise
        """
        super().__init__()
        self.exception = exception
        self.error = error
        self.__ty = ty
        self.__transfer = transfer

    def __str__(self) -> str:
        return f"AlwaysThrow[{self.__transfer.name}, {self.exception}({repr(self.exception)}) → {self.__ty}"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        flow.throw_exception(self.exception, self.error)
        return self.__ty.machine_type()(None)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return ()

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__ty, self.__transfer


class CheckAndUnwind(Handle[PythonControlFlow]):
    """
    Wraps another handle, usually a function, and checks if a Python exception has been raised.

    This serves as an adapter between arbitrary function call handles that don't return useful status codes and the flow
    control mechanism.
    """
    __handle: Handle[PythonControlFlow]

    def __init__(self, handle: Handle[PythonControlFlow]):
        super().__init__()
        self.__handle = handle

    def __str__(self) -> str:
        return f"CheckAndUnwind({self.__handle})"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        return diagram.wrap(self.__handle, args)

    def generate_handle_ir(self, flow: PythonControlFlow, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        result = self.__handle.generate_handle_ir(flow, args)
        occurred_fn = flow.use_native_function("PyErr_Occurred", PY_OBJECT_TYPE.machine_type(), ())
        comparison = flow.builder.icmp_unsigned('==',
                                                flow.builder.call(occurred_fn, ()),
                                                PY_OBJECT_TYPE.machine_type()(None))
        fail_block = flow.builder.append_basic_block('check_failed')
        success_block = flow.builder.append_basic_block('check_ok')
        flow.builder.cbranch(comparison, success_block, fail_block)
        flow.builder.position_at_start(fail_block)
        flow.unwind()
        flow.builder.position_at_start(success_block)
        return result

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return self.__handle.handle_arguments()

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__handle.handle_return()


class CFunctionHandle(BaseIndirectFunction):
    """
    Wraps a ``ctypes`` generated function in a handle
    """

    def __init__(self,
                 return_type: Type,
                 return_transfer: ReturnManagement,
                 cfunc,
                 *args: Tuple[Type, ArgumentManagement]):
        """
        Creates a new wrapper around a ctype function.

        There is not necessarily enough information in ctypes to accurate check that the function is being called with
        the correct argument types. This is some very C unsafe behaviour. YOLO.

        :param ret: the return type of this function
        :param cfunc: the ctypes function to call
        :param args: the parameters types to the function
        """
        super().__init__(return_type, return_transfer, *args)
        self.__cfunc = cfunc

    def _address(self) -> ctypes.c_size_t:
        return ctypes.c_size_t(ctypes.cast(self.__cfunc, ctypes.c_void_p).value)

    def _name(self) -> str:
        return str(self.__cfunc)


class ThrowIfNull(BaseTransferUnaryHandle[PythonControlFlow]):
    """
    A handle that throws an exception if the value is null.
    """
    error: str

    def __init__(self,
                 ty: Type,
                 transfer: ReturnManagement = ReturnManagement.BORROW,
                 error: str = "Value cannot be null"):
        """
        Create a new handle which checks if the argument is not null and raises an exception if it is null.

        :param ty: the type, which must have an LLVM type that is a pointer
        :param error: the error message to raise
        """
        super().__init__(ty, ty, transfer)
        self.error = error
        assert isinstance(ty.machine_type(), llvmlite.ir.PointerType), "Type must be a pointer"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        (arg,) = args
        return diagram.transform(arg, "Throw If Null")

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        (value, ) = args
        fail_block = flow.builder.append_basic_block('py_non_null_fail')
        success_block = flow.builder.append_basic_block('py_non_null_success')
        comparison = flow.builder.icmp_unsigned('==', value, self.handle_return()[0].machine_type()(None))
        flow.builder.cbranch(comparison, fail_block, success_block)
        flow.builder.position_at_start(fail_block)
        flow.throw_exception("ValueError", self.error)
        flow.builder.position_at_start(success_block)
        return value

    def _name(self) -> str:
        return "ThrowIfNull"


class CheckedCast(BaseTransferUnaryHandle[PythonControlFlow]):
    """
    Check a Python type is an instance of a class provided

    This is meant to be a type assertion. It's named for the JVM ``CHECKEDCAST`` instruction, which Python doesn't have.
    """

    def __init__(self,
                 arg: Union[type, PyObjectType],
                 ret: Union[type, PyObjectType],
                 transfer: ReturnManagement = ReturnManagement.BORROW):
        """
        Create a new check handle.

        :param arg: the argument type as a Python type (not a dispyatcher or LLVM type)
        :param ret: the return type as a Python type (not a dispyatcher or LLVM type)
        """
        super().__init__(
            PyObjectType(ret) if isinstance(ret, type) else ret,
            PyObjectType(arg) if isinstance(arg, type) else arg,
            transfer)

    def _name(self) -> str:
        return "CheckedCast"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        (arg,) = args
        return diagram.transform(arg, f"→ {self.handle_return()[0].python_type}")

    def generate_handle_ir(self, flow: PythonControlFlow, args: Sequence[IRValue]) -> IRValue:
        arg_type: PyObjectType
        ret_type: PyObjectType
        (arg,) = args
        (ret_type, _) = self.handle_return()
        (arg_type, _), = self.handle_arguments()

        return_type = flow.upsert_global_binding(flow.builder.module.get_unique_name("checked_cast"),
                                                 PY_OBJECT_TYPE.machine_type(),
                                                 _ptr_to_obj(ret_type.python_type))
        check = flow.use_native_function("PyObject_IsInstance",
                                         INT_RESULT_TYPE,
                                         (PY_OBJECT_TYPE.machine_type(), PY_OBJECT_TYPE.machine_type()))
        msg = f"Value of type {arg_type.python_type.__qualname__} cannot be cast to {ret_type.python_type.__qualname__}"
        flow.check_return_code(flow.builder.call(check, (arg, flow.builder.load(return_type))),
                               1,
                               "TypeError",
                               msg)
        return arg


class BooleanResultHandle(Handle[PythonControlFlow]):
    """
    A handle that takes an integer value that should be Boolean, but can be -1 for an exception
    """
    __arg: Type

    def __init__(self, input_type: Type):
        """
        Creates a new Boolean result handle.

        :param input_type: the result type (normally a platform-specific sized int)
        """
        super().__init__()
        self.__arg = input_type
        assert isinstance(input_type.machine_type(), llvmlite.ir.IntType),\
            f"Boolean result check requires an integer type, but {input_type} provided"

    def __str__(self) -> str:
        return f"BooleanResult({self.__arg}) → bool"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        (arg,) = args
        return diagram.transform(arg, 'Result → Boolean')

    def generate_handle_ir(self, flow: PythonControlFlow, args: Sequence[IRValue]) -> IRValue:
        (result,) = args
        exception_block = flow.builder.append_basic_block("result_exception")
        bool_block = flow.builder.append_basic_block("result_bool")
        flow.builder.cbranch(flow.builder.icmp_signed("<=", result, llvmlite.ir.Constant(INT_RESULT_TYPE, -1)),
                             exception_block, bool_block)

        flow.builder.position_at_start(exception_block)
        flow.unwind()

        flow.builder.position_at_start(bool_block)

        return flow.builder.icmp_signed('!=', result, llvmlite.ir.Constant(INT_RESULT_TYPE, 0))

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return (self.__arg, ArgumentManagement.BORROW_TRANSIENT),

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return MachineType(llvmlite.ir.IntType(1)), ReturnManagement.TRANSFER


class TupleArguments(NamedTuple):
    """
    The LLVM IR values needed to unpack a tuple

    Some arguments must be pre-allocated and passed to the ``PyArg_ParseTuple`` function and some must be passed to the
    handle that will consume the results, if successful.
    """
    unpack_args: Sequence[llvmlite.ir.Value]
    call_args: Callable[[], Sequence[llvmlite.ir.Value]]


class TupleElement:
    """ A representation of a possible type inside a tuple that ``PyArg_ParseTuple`` can use"""

    def format_code(self) -> str:
        """The code that directs the `packing/unpacking of a tuple
        <https://septatrix.github.io/cpython-dark-docs/c-api/arg.html#parsing-arguments>`_."""
        pass

    def pack(self) -> int:
        """
        The number of input arguments that will be consumed when packing.

        :return: the argument count
        """
        pass

    def unpack(self, builder: IRBuilder) -> TupleArguments:
        """
        Prepares a tuple for unpacking

        This creates allocations for the arguments to a tuple unpacking and a callback that allows generating loading
        those arguments. As a general rule, tuple unpacking requires a pointer to a storage location (one that will
        likely be generated by ``alloca`` and then loading that location so it can be used in the target handle. This
        function creates an instructions for the ``alloca`` instructions and then provides a callback to load them when
        required, in the same builder. Some single tuple elements translate to multiple real arguments (*e.g.*, data +
        length for buffers) and a tuple element represents on logical element even if it requires multiple arguments.

        :param builder:  the LLVM IR builder in which to generate code
        :return: the argument information required
        """
        pass


class SimpleTupleElement(TupleElement):
    """
    A naive tuple element that has a 1:1 correspondence between element and argument and can be created by an ``alloc``.
    """
    __format: str
    __type: llvmlite.ir.Type

    def __init__(self, format_code: str, ty: llvmlite.ir.Type):
        """
        Create a new naive tuple element.

        :param format_code: the Python format code to use
        :param ty: the type of the element; the machine type must match the format code, but it will not be checked
        """
        super().__init__()
        self.__format = format_code
        self.__type = ty

    def format_code(self) -> str:
        return self.__format

    def pack(self) -> int:
        return 1

    def unpack(self, builder: IRBuilder) -> TupleArguments:
        alloc = builder.alloca(self.__type)
        return TupleArguments(unpack_args=(alloc,), call_args=lambda: (builder.load(alloc),))


class TupleUnpackingDispatcher(RepackingDispatcher[None]):
    """
    A dispatcher that knows how to unpack a Python tuple into multiple arguments
    """
    def __init__(self, fallback: Handle, tuple_index: int):
        """
        Construct a new handle.

        :param fallback: the handle to use if the input does not match
        :param tuple_index: the index in the fallback argument list where the tuple, as a ``PY_OBJECT_TYPE``, will occur
        """
        super().__init__(fallback, tuple_index)
        assert isinstance(fallback.handle_arguments()[tuple_index][0], PyObjectType),\
            "First common argument is not a Python object"

    def _find_repack(self,
                     input_args: Sequence[Tuple[Type, ArgumentManagement]],
                     output_args: Sequence[Tuple[Type, ArgumentManagement]],
                     hint: None) -> Union[Repacker, None]:
        tuple_args = [find_unpack(arg) for arg, _ in output_args[0:len(output_args) - len(input_args) + 1]]
        return _TupleUnpacker(tuple_args)


class _TupleUnpacker(Repacker):
    __elements: List[TupleElement]

    def __init__(self, element: List[TupleElement]):
        super().__init__()
        self.__elements = element

    def input_count(self) -> int:
        return 1

    def output_count(self) -> int:
        return len(self.__elements)

    def generate_ir(self, state: RepackingState, args: Sequence[IRValue]) -> Sequence[Tuple[IRValue, Set[int]]]:
        i8 = llvmlite.ir.IntType(8)
        i32 = llvmlite.ir.IntType(32)
        (arg, ) = args
        outputs = []
        fmt_code = "".join(element.format_code() for element in self.__elements).encode("utf-8") + b"\x00"
        fmt_const = llvmlite.ir.Constant(llvmlite.ir.ArrayType(i8, len(fmt_code)), bytearray(fmt_code))
        fmt_global = llvmlite.ir.GlobalVariable(state.flow.builder.module, fmt_const.type,
                                                state.flow.builder.module.get_unique_name("tuple_format"))
        fmt_global.initializer = fmt_const
        fmt_global.global_constant = True
        unpack_args = [arg, state.flow.builder.bitcast(fmt_global, i8.as_pointer())]
        for element in self.__elements:
            element_args = element.unpack(state.flow.builder)
            outputs.append(element_args.call_args)
            unpack_args.extend(element_args.unpack_args)

        fn = state.flow.use_native_function("PyArg_ParseTuple", i32, [PY_OBJECT_TYPE.machine_type(), i8.as_pointer()],
                                            var_arg=True)
        unpack_result = state.flow.builder.call(fn, unpack_args)
        clear_fn = state.flow.use_native_function("PyErr_Clear", llvmlite.ir.VoidType(), [])
        state.flow.builder.call(clear_fn, [])
        state.alternate_on_bool(state.flow.builder.icmp_unsigned("!=", unpack_result, i32(0)))

        output_values = [(o, {0}) for output in outputs for o in output()]
        return output_values

    def __str__(self) -> str:
        return "Tuple " + "".join(element.format_code() for element in self.__elements)


INT_TUPLE_ELEMENT = {
    8: SimpleTupleElement("b", llvmlite.ir.IntType(8)),
    16: SimpleTupleElement("h", llvmlite.ir.IntType(16)),
    32: SimpleTupleElement("i", llvmlite.ir.IntType(32)),
    64: SimpleTupleElement("l", llvmlite.ir.IntType(64)),
    128: SimpleTupleElement("L", llvmlite.ir.IntType(128)),
}
INT_TUPLE_ELEMENT_ALLOW_OVERFLOW = {
    8: SimpleTupleElement("B", llvmlite.ir.IntType(8)),
    16: SimpleTupleElement("H", llvmlite.ir.IntType(16)),
    32: SimpleTupleElement("I", llvmlite.ir.IntType(32)),
    64: SimpleTupleElement("k", llvmlite.ir.IntType(64)),
    128: SimpleTupleElement("K", llvmlite.ir.IntType(128)),
}

CHAR_TUPLE_ELEMENT = SimpleTupleElement("c", llvmlite.ir.IntType(8))
UNICHAR_TUPLE_ELEMENT = SimpleTupleElement("C", llvmlite.ir.IntType(32))
FLOAT_TUPLE_ELEMENT = SimpleTupleElement("f", llvmlite.ir.FloatType())
DOUBLE_TUPLE_ELEMENT = SimpleTupleElement("d", llvmlite.ir.DoubleType())

PY_OBJECT_TUPLE_ELEMENT = SimpleTupleElement("O", PY_OBJECT_TYPE.machine_type())


class _PyCComplexType(Type, TupleElement):

    def __str__(self) -> str:
        return "PyComplex"

    def clone(self, flow: "F", value: IRValue) -> IRValue:
        return value

    def clone_is_self_contained(self) -> bool:
        return True

    def ctypes_type(self):
        return None

    def drop(self, flow: "FlowState", value: IRValue) -> None:
        pass

    def format_code(self) -> str:
        return "D"

    def machine_type(self) -> LLType:
        return llvmlite.ir.LiteralStructType((llvmlite.ir.DoubleType(), llvmlite.ir.DoubleType()))

    def pack(self) -> int:
        return 1

    def unpack(self, builder: IRBuilder) -> TupleArguments:
        alloc = builder.alloca(self.machine_type())
        return TupleArguments(unpack_args=(alloc,), call_args=lambda: (builder.load(alloc),))

    def into_type(self, target) -> Optional["Handle"]:
        if isinstance(target, PyObjectType) and isinstance(complex, target.python_type):
            return PY_COMPLEX_FROM_C // target
        return None

    def from_type(self, source) -> Optional["Handle"]:
        if isinstance(source, PyObjectType):
            return PY_COMPLEX_TO_C / (source,)
        return None


class NullToNone(BaseTransferUnaryHandle[PythonControlFlow]):

    def __init__(self, ty: Union[type, PyObjectType], transfer: ReturnManagement = ReturnManagement.BORROW):
        t = ty if isinstance(ty, PyObjectType) else PyObjectType(ty)
        super().__init__(t, t, transfer)

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        (arg,) = args
        return diagram.transform(arg, 'Null → None')

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        (arg,) = args
        (ty, _) = self.handle_return()
        none_block = flow.builder.append_basic_block("none")
        end_block = flow.builder.append_basic_block("end")
        original_block = flow.builder.block
        is_some = flow.builder.icmp_unsigned('!=', arg, ty.machine_type()(None))
        flow.builder.cbranch(is_some, end_block, none_block)

        flow.builder.position_at_start(none_block)

        none = flow.use_native_global("_Py_NoneStruct", ty.machine_type().pointee)
        if self.return_management == ReturnManagement.TRANSFER:
            inc_ref = flow.use_native_function("Py_IncRef", llvmlite.ir.VoidType(), (PY_OBJECT_TYPE.machine_type(),))
            flow.builder.call(inc_ref, (none,))
        flow.builder.branch(end_block)
        flow.builder.position_at_start(end_block)
        result = flow.builder.phi(ty.machine_type(), "none_to_null")
        result.add_incoming(arg, original_block)
        result.add_incoming(none, none_block)
        return result

    def _name(self) -> str:
        return "NullToNone"


PY_CCOMPLEX_TYPE = _PyCComplexType()
""" The type for complex number """


PY_CCOMPLEX_REAL = GetElementPointer(PY_CCOMPLEX_TYPE, 0,
                                     dispyatcher.Pointer(MachineType(llvmlite.ir.DoubleType())))
PY_CCOMPLEX_IMAG = GetElementPointer(PY_CCOMPLEX_TYPE, 1,
                                     dispyatcher.Pointer(MachineType(llvmlite.ir.DoubleType())))


def implode_complex_number(handle: Handle, index: int) -> Handle:
    return implode_args(handle, index, PY_CCOMPLEX_REAL + DerefPointer, PY_CCOMPLEX_IMAG + DerefPointer)


def find_unpack(ty: Type) -> TupleElement:
    """
    Attempts to find a tuple element that matches an arbitrary type

    If the type implements ``TupleElement``, then it will be used as itself. Otherwise, if the type has a numeric
    machine type, an appropriate tuple element based on that machine type will be used.

    :param ty: the type to find a matching tuple element representation of
    :return: the best match, if found, otherwise, a ``ValueError`` will be raised.
    """
    if isinstance(ty, TupleElement):
        return ty
    if ty == PY_OBJECT_TYPE:
        return PY_OBJECT_TUPLE_ELEMENT
    if isinstance(ty.machine_type(), llvmlite.ir.IntType):
        return INT_TUPLE_ELEMENT[ty.machine_type().width]
    if isinstance(ty.machine_type(), llvmlite.ir.FloatType):
        return FLOAT_TUPLE_ELEMENT
    if isinstance(ty.machine_type(), llvmlite.ir.DoubleType):
        return DOUBLE_TUPLE_ELEMENT
    raise ValueError(f"No tuple conversion for {ty}")


class WithGlobalInterpreterLock(Handle[PythonControlFlow]):
    """
    Executes the inner handle while holding the Python global interpreter lock
    """
    __handle: Handle

    def __init__(self, handle: Handle):
        super().__init__()
        self.__handle = handle

    def __str__(self) -> str:
        return f"WithGIL[{self.__handle}]"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        return diagram.wrap(self.__handle, args)

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        state = flow.builder.call(flow.use_native_function("PyGILState_Ensure", INT_RESULT_TYPE, ()))
        release = flow.use_native_function("PyGILState_Release", llvmlite.ir.VoidType(), (INT_RESULT_TYPE,))
        flow.unwind_cleanup(lambda: flow.builder.call(release, state))
        return self.__handle.generate_handle_ir(flow, args)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return self.__handle.handle_arguments()

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__handle.handle_return()


class WithoutGlobalInterpreterLock(Handle[PythonControlFlow]):
    """
    Executes the inner handle while releasing the Python global interpreter lock
    """
    __handle: Handle[ControlFlow]

    def __init__(self, handle: Handle[ControlFlow]):
        super().__init__()
        self.__handle = handle

    def __str__(self) -> str:
        return f"WithoutGIL[{self.__handle}]"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        return diagram.wrap(self.__handle, args)

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        thread_state_type = llvmlite.ir.IntType(8).as_pointer()
        state = flow.builder.call(flow.use_native_function("PyEval_SaveThread", thread_state_type, ()))
        release = flow.use_native_function("PyEval_RestoreThread", llvmlite.ir.VoidType(), (thread_state_type,))
        flow.unwind_cleanup(lambda: flow.builder.call(release, state))
        return self.__handle.generate_handle_ir(flow, args)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return self.__handle.handle_arguments()

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__handle.handle_return()


class Value(Handle[PythonControlFlow]):
    """
    A handle that returns a reference to a Python object
    """
    __type: type
    __value: Any

    def __init__(self,
                 value: Any,
                 ty: Union[type, PyObjectType, None] = None):
        """
        Creates a new handle that returns an updatable Python object

        :param value: the value to return
        :param ty: the Python type the handle will return. This can be explicitly provided as a Python type or a
            ``PyObjectType`` or it can be automatically inferred from the value provided (``None``). Note that the value
            must be an instance of this type.
        """
        super().__init__()
        if ty is None:
            self.__type = type(value)
        elif isinstance(ty, PyObjectType):
            self.__type = ty.python_type
        elif isinstance(ty, type):
            self.__type = ty
        else:
            raise TypeError(f"Cannot create handle for type {ty}.")
        assert isinstance(value, self.__type), f"Value {value} is not of type {self.__type}"
        self.__value = value

    @property
    def value(self) -> Any:
        return self.__value

    @value.setter
    def value(self, value) -> None:
        assert isinstance(value, self.__type), f"Value {value} is not of type {self.__type}"
        self.__value = value
        self.invalidate_address(f"value_{id(self)}", _ptr_to_obj(self.__value))

    def __str__(self) -> str:
        return f"Value({repr(self.__value)}) → {self.__type}"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        global_value = flow.upsert_global_binding(f"value_{id(self)}",
                                                  PY_OBJECT_TYPE.machine_type(),
                                                  _ptr_to_obj(self.__value))
        return flow.builder.load(global_value)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return ()

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return PyObjectType(self.__type), ReturnManagement.BORROW


def callback(return_type: Type, *arguments: Type) -> Callable[[Callable], Handle[PythonControlFlow]]:
    """
    Generates a constructor for callbacks of a certain type.

    This allows exporting a Python function as a handle. It assumes the handle will be invoked within a Python control
    flow.

    :param return_type: the return type of the function
    :param arguments:
    :return:
    """
    func_type = ctypes.PYFUNCTYPE(return_type.ctypes_type(), *(a.ctypes_type() for a in arguments))

    class CallbackHandle(BaseIndirectFunction):
        __func: func_type

        def __init__(self, callback: Callable):
            super().__init__(return_type,
                             ReturnManagement.TRANSFER,
                             *((a, ArgumentManagement.BORROW_TRANSIENT) for a in arguments))
            self.__func = func_type(callback)

        def _name(self) -> str:
            return str(self.__func)

        def _address(self) -> ctypes.c_size_t:
            return ctypes.c_size_t(ctypes.cast(self.__func, ctypes.c_void_p).value)

    return lambda function: CallbackHandle(function) @ CheckAndUnwind


PY_DICT_NEW = CurrentProcessFunction(PY_DICT_TYPE, ReturnManagement.TRANSFER, "PyDict_New")
PY_DICT_CLEAR = CurrentProcessFunction(MachineType(llvmlite.ir.VoidType()),
                                       ReturnManagement.TRANSFER,
                                       "PyDict_Clear",
                                       (PY_DICT_TYPE, ArgumentManagement.BORROW_TRANSIENT))
PY_DICT_CONTAINS = CurrentProcessFunction(
    MachineType(INT_RESULT_TYPE),
    ReturnManagement.TRANSFER,
    "PyDict_Contains",
    (PY_DICT_TYPE, ArgumentManagement.BORROW_TRANSIENT),
    (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT)) + BooleanResultHandle
PY_DICT_COPY = CurrentProcessFunction(PY_DICT_TYPE,
                                      ReturnManagement.TRANSFER,
                                      "PyDict_Copy",
                                      (PY_DICT_TYPE, ArgumentManagement.BORROW_TRANSIENT))
PY_DICT_GET_ITEM = CurrentProcessFunction(PY_OBJECT_TYPE,
                                          ReturnManagement.BORROW,
                                          "PyDict_GetItem",
                                          (PY_DICT_TYPE, ArgumentManagement.BORROW_CAPTURE),
                                          (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT))
PY_DICT_GET_ITEM_STRING = CurrentProcessFunction(PY_OBJECT_TYPE,
                                                 ReturnManagement.BORROW,
                                                 "PyDict_GetItemString",
                                                 (PY_DICT_TYPE, ArgumentManagement.BORROW_CAPTURE),
                                                 (CHAR_ARRAY_TYPE, ArgumentManagement.BORROW_TRANSIENT))
PY_DICT_SIZE = CurrentProcessFunction(MachineType(SIZE_T_TYPE),
                                      ReturnManagement.TRANSFER,
                                      "PyDict_Size",
                                      (PY_DICT_TYPE, ArgumentManagement.BORROW_TRANSIENT))
PY_UNICODE_FROM_STRING = CurrentProcessFunction(PyObjectType(str),
                                                ReturnManagement.TRANSFER,
                                                "PyUnicode_FromString",
                                                (CHAR_ARRAY_TYPE, ArgumentManagement.BORROW_TRANSIENT))

PY_OBJECT_GET_ATTR = CurrentProcessFunction(PY_OBJECT_TYPE,
                                            ReturnManagement.TRANSFER,
                                            "PyObject_GetAttr",
                                            (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT),
                                            (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT))

PY_OBJECT_GET_ATTR_STRING = CurrentProcessFunction(PY_OBJECT_TYPE,
                                                   ReturnManagement.TRANSFER,
                                                   "PyObject_GetAttrString",
                                                   (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT),
                                                   (CHAR_ARRAY_TYPE, ArgumentManagement.BORROW_TRANSIENT))

PY_OBJECT_GET_ITEM = CurrentProcessFunction(PY_OBJECT_TYPE,
                                            ReturnManagement.TRANSFER,
                                            "PyObject_GetItem",
                                            (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT),
                                            (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT))

PY_FLOAT_AS_DOUBLE = CurrentProcessFunction(MachineType(llvmlite.ir.DoubleType()),
                                            ReturnManagement.TRANSFER,
                                            "PyFloat_AsDouble",
                                            (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT)) @ CheckAndUnwind

PY_LONG_AS_LONG = CurrentProcessFunction(MachineType(llvmlite.ir.IntType(ctypes.sizeof(ctypes.c_long) * 8)),
                                         ReturnManagement.TRANSFER,
                                         "PyLong_AsLong",
                                         (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT)) @ CheckAndUnwind

PY_LONG_FROM_LONG = CurrentProcessFunction(PyObjectType(int),
                                           ReturnManagement.TRANSFER,
                                           "PyLong_AsLong",
                                           (LONG_TYPE, ArgumentManagement.BORROW_TRANSIENT)) + ThrowIfNull

PY_COMPLEX_REAL_AS_DOUBLE = CurrentProcessFunction(MachineType(llvmlite.ir.DoubleType()),
                                                   ReturnManagement.TRANSFER,
                                                   "PyComplex_RealAsDouble",
                                                   (PyObjectType(complex), ArgumentManagement.BORROW_TRANSIENT))\
                            @ CheckAndUnwind

PY_COMPLEX_IMAG_AS_DOUBLE = CurrentProcessFunction(MachineType(llvmlite.ir.DoubleType()),
                                                   ReturnManagement.TRANSFER,
                                                   "PyComplex_ImagAsDouble",
                                                   (PyObjectType(complex), ArgumentManagement.BORROW_TRANSIENT))\
                            @ CheckAndUnwind

PY_COMPLEX_TO_C = CurrentProcessFunction(PY_CCOMPLEX_TYPE,
                                         ReturnManagement.TRANSFER,
                                         "PyComplex_AsCComplex",
                                         (PY_OBJECT_TYPE, ArgumentManagement.BORROW_TRANSIENT)) @ CheckAndUnwind

PY_COMPLEX_FROM_C = CurrentProcessFunction(PyObjectType(complex),
                                           ReturnManagement.TRANSFER,
                                           "PyComplex_FromCComplex",
                                           (PY_CCOMPLEX_TYPE, ArgumentManagement.BORROW_TRANSIENT))
