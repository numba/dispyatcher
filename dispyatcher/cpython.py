import ctypes
from typing import Sequence, Dict, Union, List, NamedTuple, Callable

import llvmlite.ir
from llvmlite.ir import IRBuilder, Value as IRValue, Type as LLType

import dispyatcher.general
import llvmlite.ir.types
from dispyatcher import Type, Handle, llvm_type_to_ctype, F, IdentityHandle
from dispyatcher.accessors import GetElementPointer
from dispyatcher.general import MachineType, CurrentProcessFunctionHandle
from dispyatcher.permute import implode_args
from dispyatcher.repacking import RepackingDispatcher, Repacker, RepackingFlow
from dispyatcher.resource import ResourceHandle

INT_RESULT_TYPE = llvmlite.ir.types.IntType(ctypes.sizeof(ctypes.c_int) * 8)
SIZE_T_TYPE = llvmlite.ir.types.IntType(ctypes.sizeof(ctypes.c_size_t) * 8)


class PythonControlFlow(dispyatcher.ControlFlow):
    """
    A control flow that knows how to generate Python exceptions
    """
    __return_type: LLType

    def __init__(self, builder: IRBuilder, return_type: LLType):
        super().__init__(builder)
        self.__return_type = return_type

    def throw_exception(self, error: str, message: str) -> None:
        """
        Throw an exception and stop the current control flow
        :param error: the name of the exception (only the built-in exceptions such as `ValueError`)
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
        self._cleanup()
        self.builder.ret(self.__return_type(None))

    def check_return_code(self, result: IRValue, ok_value: int, error: str, message: str) -> None:
        """
        Many CPython functions return an integer for a true/false/exception case. This is a helper to build the flow
        logic for these functions
        :param result: the result returned by the function
        :param ok_value: the return value that signals the happy path
        :param error: the exception name when the function is on the unhappy non-exception path
        :param message: the message to raise on the unhappy non-exception path
        """
        fail_block = self.builder.append_basic_block("result_fail")
        ok_block = self.builder.append_basic_block("result_ok")
        exception_block = self.builder.append_basic_block("result_exception")
        self.builder.switch(result, fail_block)\
            .add_case(llvmlite.ir.Constant(INT_RESULT_TYPE, ok_value), ok_block)\
            .add_case(llvmlite.ir.Constant(INT_RESULT_TYPE, -1), exception_block)

        self.builder.position_at_start(fail_block)
        self.throw_exception(error, message)
        self.builder.position_at_start(exception_block)
        self.unwind()
        self.builder.position_at_start(ok_block)


class PythonControlFlowType(dispyatcher.ControlFlowType):
    """
    The control flow that allows CPython exceptions to be thrown
    """

    def create_flow(self, builder: IRBuilder, return_type: Type, arg_types: Sequence[Type]) -> F:
        return PythonControlFlow(builder, return_type.machine_type())

    def ctypes_function(self, return_type: Type, arg_types: Sequence[Type]):
        return ctypes.PYFUNCTYPE(return_type.ctypes_type(), *(a.ctypes_type() for a in arg_types))


class PyObjectType(Type):
    """
    The type of a Python object, using the `PyObject*` at the machine level, that also knows a Python type and can
    insert appropriate type checks during handle conversions.
    """
    __type: type

    def __init__(self, ty: type):
        self.__type = ty

    def ctypes_type(self):
        return ctypes.py_object

    def machine_type(self) -> LLType:
        return llvmlite.ir.PointerType(llvmlite.ir.IntType(8))

    def __str__(self) -> str:
        return f"PyObject({self.__type})"

    def into_type(self, target: Type) -> Union[Handle, None]:
        if isinstance(target, PyObjectType):
            if issubclass(target.__type, self.__type):
                return IdentityHandle(target, self)
            else:
                return CheckedCast(self.__type, target.__type)

    @property
    def python_type(self) -> type:
        return self.__type

    def __repr__(self) -> str:
        return f"dispyatcher.cpython.PyObjectType({repr(self.__type)})"


PY_OBJECT_TYPE = PyObjectType(object)
""" The type for an unknown Python object (in C, `PyObject*`) """

PY_DICT_TYPE = PyObjectType(dict)
""" The type for a Python dictionary """


class CFunctionHandle(Handle):
    """
    Wraps a `ctypes` generated function in a handle
    """
    __args: Sequence[Type]
    __ret: Type

    def __init__(self, ret: Type, cfunc, *args: Type, ignore_return_type=False):
        """
        Creates a new wrapper around a ctype function.

        There is not necessarily enough information in ctypes to accurate check that the function is being called with
        the correct argument types. This is some very C unsafe behaviour. If the argument types are filled out in the
        ctypes function, which they are often not, this will attempt to validate them against arguments provided. If
        no arguments are known in the ctypes function, YOLO.
        :param ret: the return type of this function; since this must be filled out, this will always be checked
        :param cfunc: the ctypes function to call
        :param args: the parameters types to the function
        """
        super().__init__()
        self.__cfunc = cfunc
        self.__args = args
        self.__ret = ret
        if not ignore_return_type:
            machine_restype = llvm_type_to_ctype(ret.machine_type())
            assert machine_restype == cfunc.restype,\
                f"ctype function result {cfunc.restype} not compatible with specified return type {machine_restype}"
        if cfunc.argtypes:
            assert len(cfunc.argtypes) == len(args),\
                f"Argument lengths don't match. Got {len(args)}, ctypes expected {len(cfunc.argtypes)}."
            for index, (cfunc_arg, arg_type) in enumerate(zip(cfunc.argtypes, args)):
                assert llvm_type_to_ctype(arg_type.machine_type()) == cfunc_arg, \
                    f"Argument {index} is expected to be {cfunc_arg}, but got {arg_type}."

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__ret, self.__args

    def generate_ir(self, flow: dispyatcher.F, args: Sequence[IRValue]) -> IRValue:
        fn_name = flow.builder.module.get_unique_name("cfunc")
        fn_type = llvmlite.ir.FunctionType(self.__ret.machine_type(), [arg.machine_type() for arg in self.__args])
        fn = flow.upsert_global_binding(fn_name, fn_type, ctypes.c_char_p.from_address(ctypes.addressof(self.__cfunc)))
        return flow.builder.call(flow.builder.load(fn), args)

    def __str__(self) -> str:
        return f"CType {self.__cfunc}({', '.join(str(a) for a in self.__args)}) → {self.__ret}"


class ThrowIfNull(Handle):
    """
    A handle that throws an exception if the value is null.
    """
    error: str
    __type: Type

    def __init__(self, ty: Type, error: str = "Value cannot be null"):
        """
        Create a new handle which checks if the argument is not null and raises an exception if it is null.
        :param ty: the type, which must have an LLVM type that is a pointer
        :param error: the error message to raise
        """
        super().__init__()
        self.__type = ty
        self.error = error
        assert isinstance(ty.machine_type(), llvmlite.ir.PointerType), "Type must be a pointer"

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__type, (self.__type,)

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        (value, ) = args
        fail_block = flow.builder.append_basic_block('py_non_null_fail')
        success_block = flow.builder.append_basic_block('py_non_null_success')
        comparison = flow.builder.icmp_unsigned('==', value, self.__type.machine_type()(None))
        flow.builder.cbranch(comparison, fail_block, success_block)
        flow.builder.position_at_start(fail_block)
        flow.throw_exception("ValueError", self.error)
        flow.builder.position_at_start(success_block)
        return value

    def __str__(self) -> str:
        return f"ThrowIfNull({self.__type}, {self.error})"


class CheckedCast(Handle[PythonControlFlow]):
    """
    Check a Python type is an instance of a class provided

    This is meant to be a type assertion. It's named for the JVM `CHECKEDCAST` instruction, which Python doesn't have.
    """
    __ret: type
    __arg: type

    def __init__(self, arg: type, ret: type):
        """
        Create a new check handle
        :param arg: the argument type as a Python type (not a dispyatcher or LLVM type)
        :param ret: the return type as a Python type (not a dispyatcher or LLVM type)
        """
        super().__init__()
        self.__ret = ret
        self.__arg = arg

    def function_type(self) -> (Type, Sequence[Type]):
        return PyObjectType(self.__ret), (PyObjectType(self.__arg),)

    def generate_ir(self, flow: PythonControlFlow, args: Sequence[IRValue]) -> IRValue:
        (arg,) = args
        return_type = flow.upsert_global_binding(flow.builder.module.get_unique_name("checked_cast"),
                                                 PY_OBJECT_TYPE.machine_type(),
                                                 ctypes.c_char_p.from_address(
                                                     ctypes.addressof(ctypes.py_object(self.__ret))))
        check = flow.use_native_function("PyObject_IsInstance",
                                         INT_RESULT_TYPE,
                                         (PY_OBJECT_TYPE.machine_type(), PY_OBJECT_TYPE.machine_type()))
        flow.check_return_code(flow.builder.call(flow.builder.load(check), (arg, return_type)),
                               1, "TypeError", f"Value of type {self.__arg} cannot be cast to {self.__ret}.")
        return arg

    def __str__(self) -> str:
        return f"CheckedCast({self.__arg} → {self.__ret})"


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

    def function_type(self) -> (Type, Sequence[Type]):
        return MachineType(llvmlite.ir.IntType(1)), (self.__arg,)

    def generate_ir(self, flow: PythonControlFlow, args: Sequence[IRValue]) -> IRValue:
        (result,) = args
        exception_block = flow.builder.append_basic_block("result_exception")
        bool_block = flow.builder.append_basic_block("result_bool")
        flow.builder.cbranch(flow.builder.icmp_signed("<=", result, llvmlite.ir.Constant(INT_RESULT_TYPE, -1)),
                             exception_block, bool_block)

        flow.builder.position_at_start(exception_block)
        flow.unwind()

        flow.builder.position_at_start(bool_block)

        return flow.builder.icmp_signed('!=', result, llvmlite.ir.Constant(INT_RESULT_TYPE, 0))

    def __str__(self) -> str:
        return f"BooleanResult {self.__arg}"


class TupleArguments(NamedTuple):
    """
    The LLVM IR values needed to unpack a tuple

    Some arguments must be pre-allocated and passed to the `PyArg_ParseTuple` function and some must be passed to the
    handle that will consume the results, if successful.
    """
    unpack_args: Sequence[llvmlite.ir.Value]
    call_args: Callable[[], Sequence[llvmlite.ir.Value]]


class TupleElement:
    """ A representation of a possible type inside a tuple that `PyArg_ParseTuple` can use"""

    def format_code(self) -> str:
        """The code that directs the `packing/unpacking of a tuple
        <https://septatrix.github.io/cpython-dark-docs/c-api/arg.html#parsing-arguments>`_."""
        pass

    def pack(self) -> int:
        """
        The number of input arguments that will be consumed when packing
        :return: the argument count
        """
        pass

    def unpack(self, builder: IRBuilder) -> TupleArguments:
        """
        Prepares a tuple for unpacking

        This creates allocations for the arguments to a tuple unpacking and a callback that allows generating loading
        those arguments. As a general rule, tuple unpacking requires a pointer to a storage location (one that will
        likely be generated by `alloca` and then loading that location so it can be used in the target handle. This
        function creates an instructions for the `alloca` instructions and then provides a callback to load them when
        required, in the same builder. Some single tuple elements translate to multiple real arguments (_e.g._, data +
        length for buffers) and a tuple element represents on logical element even if it requires multiple arguments
        :param builder:  the LLVM IR builder in which to generate code
        :return: the argument information required
        """
        pass


class SimpleTupleElement(TupleElement):
    """
    A naive tuple element that has a 1:1 correspondence between element and argument and can be created by an `alloc`.
    """
    __format: str
    __type: llvmlite.ir.Type

    def __init__(self, format_code: str, ty: llvmlite.ir.Type):
        """
        Create a new naive tuple element
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
        Construct a new handle
        :param fallback: the handle to use if the input does not match
        :param tuple_index: the index in the fallback argument list where the tuple, as a `PY_OBJECT_TYPE`, will occur
        """
        super().__init__(fallback, tuple_index)
        assert fallback.function_type()[1][tuple_index] == PY_OBJECT_TYPE,\
            "First common argument is not a Python object"

    def _find_repack(self, input_args: Sequence[Type], output_args: Sequence[Type], hint: None)\
            -> Union[Repacker, None]:
        tuple_args = [find_unpack(arg) for arg in output_args[0:len(output_args) - len(input_args) + 1]]
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

    def generate_ir(self, flow: RepackingFlow, args: Sequence[IRValue]) -> Sequence[IRValue]:
        i8 = llvmlite.ir.IntType(8)
        i32 = llvmlite.ir.IntType(32)
        (arg, ) = args
        outputs = []
        fmt_code = "".join(element.format_code() for element in self.__elements).encode("utf-8") + b"\x00"
        fmt_const = llvmlite.ir.Constant(llvmlite.ir.ArrayType(i8, len(fmt_code)), bytearray(fmt_code))
        fmt_global = llvmlite.ir.GlobalVariable(flow.builder.module, fmt_const.type,
                                                flow.builder.module.get_unique_name("tuple_format"))
        fmt_global.initializer = fmt_const
        fmt_global.global_constant = True
        unpack_args = [arg, flow.builder.bitcast(fmt_global, i8.as_pointer())]
        for element in self.__elements:
            element_args = element.unpack(flow.builder)
            outputs.append(element_args.call_args)
            unpack_args.extend(element_args.unpack_args)

        fn = flow.use_native_function("PyArg_ParseTuple", i32, [PY_OBJECT_TYPE.machine_type(), i8.as_pointer()],
                                      var_arg=True)
        unpack_result = flow.builder.call(fn, unpack_args)
        clear_fn = flow.use_native_function("PyErr_Clear", llvmlite.ir.VoidType(), [])
        flow.builder.call(clear_fn, [])
        flow.alternate_on_bool(flow.builder.icmp_unsigned("!=", unpack_result, i32(0)))

        output_values = [o for output in outputs for o in output()]
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


class _PyComplexType(Type, TupleElement):

    def ctypes_type(self):
        return None

    def machine_type(self) -> LLType:
        return llvmlite.ir.LiteralStructType((llvmlite.ir.DoubleType(), llvmlite.ir.DoubleType()))

    def __str__(self) -> str:
        return "PyComplex"

    def format_code(self) -> str:
        return "D"

    def pack(self) -> int:
        return 1

    def unpack(self, builder: IRBuilder) -> TupleArguments:
        alloc = builder.alloca(self.machine_type())
        return TupleArguments(unpack_args=(alloc,), call_args=lambda: (builder.load(alloc),))


class IncrementReference(Handle):
    __type: Type

    def __init__(self, ty: Union[type, PyObjectType]):
        super().__init__()
        if isinstance(ty, PyObjectType):
            self.__type = ty
        elif isinstance(ty, type):
            self.__type = PyObjectType(ty)
        else:
            raise TypeError(f"Don't know how to build a reference count change for {ty}")

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__type, (self.__type,)

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        (arg, ) = args
        inc_block = flow.builder.append_basic_block("incref_nonnull")
        end_block = flow.builder.append_basic_block("incref_end")
        is_some = flow.builder.icmp_unsigned('!=', arg, self.__type.machine_type()(None))
        flow.builder.cbranch(is_some, inc_block, end_block)

        flow.builder.position_at_start(inc_block)

        inc_ref = flow.use_native_function("Py_IncRef", llvmlite.ir.VoidType(), (PY_OBJECT_TYPE.machine_type(),))
        flow.builder.call(inc_ref, (arg,))
        flow.builder.branch(end_block)
        flow.builder.position_at_start(end_block)
        return arg

    def __str__(self) -> str:
        return f"IncRef({self.__type})"


class NullToNone(Handle):
    __type: Type

    def __init__(self, ty: Union[type, PyObjectType]):
        super().__init__()
        if isinstance(ty, PyObjectType):
            self.__type = ty
        elif isinstance(ty, type):
            self.__type = PyObjectType(ty)
        else:
            raise TypeError(f"Don't know how to build a reference count change for {ty}")

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__type, (self.__type,)

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        (arg,) = args
        none_block = flow.builder.append_basic_block("none")
        end_block = flow.builder.append_basic_block("end")
        original_block = flow.builder.block
        is_some = flow.builder.icmp_unsigned('!=', arg, self.__type.machine_type()(None))
        flow.builder.cbranch(is_some, end_block, none_block)

        flow.builder.position_at_start(none_block)

        none = flow.use_native_global("_Py_NoneStruct", self.__type.machine_type().pointee)
        inc_ref = flow.use_native_function("Py_IncRef", llvmlite.ir.VoidType(), (PY_OBJECT_TYPE.machine_type(),))
        flow.builder.call(inc_ref, (none,))
        flow.builder.branch(end_block)
        flow.builder.position_at_start(end_block)
        result = flow.builder.phi(self.__type.machine_type(), "none_to_null")
        result.add_incoming(arg, original_block)
        result.add_incoming(none, none_block)
        return result

    def __str__(self) -> str:
        return f"NullToNone({self.__type})"


PY_COMPLEX_TYPE = _PyComplexType()
""" The type for complex number """


PY_COMPLEX_REAL = GetElementPointer(PY_COMPLEX_TYPE, 0,
                                    dispyatcher.Pointer(MachineType(llvmlite.ir.DoubleType()))).deref()
PY_COMPLEX_IMAG = GetElementPointer(PY_COMPLEX_TYPE, 1,
                                    dispyatcher.Pointer(MachineType(llvmlite.ir.DoubleType()))).deref()


def implode_complex_number(handle: Handle, index: int) -> Handle:
    return implode_args(handle, index, PY_COMPLEX_REAL, PY_COMPLEX_IMAG)


def find_unpack(ty: Type) -> TupleElement:
    """
    Attempts to find a tuple element that matches an arbitrary type

    If the type implements `TupleElement`, then it will be used as itself. Otherwise, if the type has a numeric machine
    type, an appropriate tuple element based on that machine type will be used.
    :param ty: the type to find a matching tuple element representation of
    :return: the best match, if found, otherwise, a `ValueError` will be raised.
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


class _GlobalInterpreterLockType(Type):

    def ctypes_type(self):
        raise ValueError("Absolutely refusing to leak a Python Interpreter Lock into Python. You will have a bad time.")

    def machine_type(self) -> LLType:
        return llvmlite.ir.IntType(32)

    def __str__(self) -> str:
        return "PythonGIL"


PY_DICT_NEW = CurrentProcessFunctionHandle(PY_DICT_TYPE, "PyDict_New")
PY_DICT_CLEAR = CurrentProcessFunctionHandle(PY_DICT_TYPE, "PyDict_Clear")
PY_DICT_CONTAINS = CurrentProcessFunctionHandle(MachineType(INT_RESULT_TYPE), "PyDict_Contains",
                                                PY_DICT_TYPE, PY_OBJECT_TYPE) + BooleanResultHandle
PY_DICT_COPY = CurrentProcessFunctionHandle(PY_DICT_TYPE, "PyDict_Copy", PY_DICT_TYPE)
PY_DICT_GET_ITEM = CurrentProcessFunctionHandle(PY_OBJECT_TYPE, "PyDict_GetItem", PY_DICT_TYPE, PY_OBJECT_TYPE)
PY_DICT_SIZE = CurrentProcessFunctionHandle(MachineType(SIZE_T_TYPE), "PyDict_Size", PY_DICT_TYPE)

GIL_TYPE = _GlobalInterpreterLockType()
GIL_STATE_ENSURE_HANDLE = CurrentProcessFunctionHandle(GIL_TYPE, "PyGILState_Ensure")
GIL_STATE_RELEASE_HANDLE = CurrentProcessFunctionHandle(dispyatcher.general.MachineType(llvmlite.ir.VoidType()),
                                                        "PyGILState_Release", GIL_TYPE)


def with_gil(inner: Handle) -> Handle:
    """
    Wraps a handle so that the Python GIL will be acquired before the handle is executed and released after.
    :param inner: the handle to wrap
    :return: the GIL-protected handle
    """
    return ResourceHandle(inner, None, GIL_STATE_ENSURE_HANDLE, GIL_STATE_RELEASE_HANDLE)
