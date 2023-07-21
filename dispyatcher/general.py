import ctypes
from typing import Sequence, Any, Optional, Tuple, Union

import llvmlite.ir
from llvmlite.ir import Type as LLType, Value as IRValue
from llvmlite.binding import ResourceTracker

from dispyatcher import ArgumentManagement, BaseTransferUnaryHandle, ControlFlow, Deref, F, FlowState, Handle,\
    ReturnManagement, TemporaryValue, Type, is_llvm_floating_point
from dispyatcher.accessors import GetPointer


class MachineType(Type):
    """
    A high-level type that operates on low-level machine types

    This would be the most direct mapping to C language types available.
    """
    __type: LLType

    def __init__(self, ty: LLType):
        """
        Construct a new instance of a machine type given the matching LLVM type.

        :param ty: the LLVM type
        """
        self.__type = ty

    def __eq__(self, o: object) -> bool:
        if isinstance(o, MachineType):
            return self.__type == o.__type
        else:
            return False

    def __repr__(self) -> str:
        return f"MachineType({self.__type})"

    def __str__(self) -> str:
        return str(self.__type)

    def into_type(self, target: Type) -> Optional[Handle]:
        if isinstance(target, MachineType):
            self_is_int = isinstance(self.__type, llvmlite.ir.IntType)
            self_is_fp = is_llvm_floating_point(self.__type)
            other_is_int = isinstance(target.__type, llvmlite.ir.IntType)
            other_is_fp = is_llvm_floating_point(target.__type)
            if self_is_int and other_is_int:
                return IntegerResize(self, target)
            if self_is_fp and other_is_fp:
                return FloatResize(self, target)
            if self_is_int and other_is_fp:
                return IntegerToFloat(self, target)
            if self_is_fp and other_is_int:
                return FloatToInteger(self, target)
            if isinstance(self.__type, llvmlite.ir.PointerType) and isinstance(target.__type, llvmlite.ir.PointerType):
                return BitCast(self, target)
        return None

    def from_type(self, source: Type) -> Optional[Handle]:
        return None

    def machine_type(self) -> LLType:
        return self.__type

    def clone(self, flow: F, value: IRValue) -> IRValue:
        return value

    def clone_is_self_contained(self) -> bool:
        return True

    def drop(self, flow: FlowState, value: IRValue) -> None:
        # No drop required
        return


class UncheckedArray(Deref, GetPointer):
    """
    A high-level type representing a C-like array.

    This array is a linear block of entries with no bounds checking or known length. It provides the same unsafe
    behaviour as C; you're welcome. It is intended mostly for compatibility with C.
    """
    __element_type: Type

    def __init__(self, element_type: Type):
        self.__element_type = element_type

    def __eq__(self, o: object) -> bool:
        if isinstance(o, UncheckedArray):
            return self.__element_type == o.__element_type
        else:
            return False

    def __str__(self) -> str:
        return f"UncheckedArray({self.__element_type})"

    def target(self) -> Type:
        return self.__element_type

    def target_pointer(self) -> Type:
        return self

    def ctypes_type(self):
        return ctypes.POINTER(self.__element_type.ctypes_type())

    def machine_type(self) -> LLType:
        return self.__element_type.machine_type().as_pointer()


class BitCast(BaseTransferUnaryHandle[ControlFlow]):
    """
    Performs an LLVM bit-cast to reinterpret the memory of one type as another.

    This handle assumes the type conversion is acceptable for the types provided; it does not validate that conversion
    will yield valid results.
    """
    __source: Type
    __target: Type
    __management: bool

    def __init__(self, source: Type, target: Type, transfer: ReturnManagement = ReturnManagement.BORROW):
        super().__init__(target, source, transfer)

    def _name(self) -> str:
        return "BitCast"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        (arg, ) = args
        return flow.builder.bitcast(arg, self.handle_return()[0].machine_type())


class IntegerResize(BaseTransferUnaryHandle[ControlFlow]):
    """
    Convert an integer to the correct size through signed extension or truncation, as appropriate.

    The machine types for both of the types provided must be `llvm.ir.IntType`. This handle assumes the type conversion
    is acceptable for the types provided; it only cares about the machine precision.
    """

    def __init__(self, source: Type, target: Type, transfer: ReturnManagement = ReturnManagement.BORROW):
        super().__init__(target, source, transfer)
        assert isinstance(source.machine_type(), llvmlite.ir.IntType), "Source type must have an integer machine type"
        assert isinstance(target.machine_type(), llvmlite.ir.IntType), "target type must have an integer machine type"
        self.__source = source
        self.__target = target

    def _name(self) -> str:
        return "IntegerResize"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        target, _ = self.handle_return()
        ((source, _),) = self.handle_arguments()
        (arg, ) = args
        if source.machine_type().width > target.machine_type().width:
            return flow.builder.trunc(arg, target.machine_type())
        if source.machine_type().width < target.machine_type().width:
            return flow.builder.sext(arg, target.machine_type())
        return arg


class IntegerToFloat(BaseTransferUnaryHandle[ControlFlow]):
    """
    Converts an integer type to a floating point type

    The source type must have a machine type of ``llvm.ir.IntType`` and the target must have a machine type of
    ``llvm.ir.HalfType``, ``llvm.ir.FloatType``, or ``llvm.ir.DoubleType``. This handle assumes the type conversion is
    acceptable for the types provided; it only cares about the machine precision.
    """
    __source: Type
    __target: Type

    def __init__(self, source: Type, target: Type, transfer: ReturnManagement = ReturnManagement.BORROW):
        super().__init__(target, source, transfer)
        assert isinstance(source.machine_type(), llvmlite.ir.IntType), "Source is not an integer type"
        assert is_llvm_floating_point(target.machine_type()), "Target is not a floating point type"

    def _name(self) -> str:
        return "IntegerToFloat"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        (arg, ) = args
        return flow.builder.sitofp(arg, self.handle_return()[0].machine_type())


class FloatResize(BaseTransferUnaryHandle[ControlFlow]):
    """
    Convert a floating-point to the correct size through extension or truncation, as appropriate.

    The machine types for both of the types provided must be ``llvm.ir.HalfType``, ``llvm.ir.FloatType``, or
    ``llvm.ir.DoubleType``. This handle assumes the type conversion is acceptable for the types provided; it only cares
    about the machine precision.
    """

    def __init__(self, source: Type, target: Type, transfer: ReturnManagement = ReturnManagement.BORROW):
        super().__init__(target, source, transfer)
        assert is_llvm_floating_point(source.machine_type()), "Source type must have a floating-point machine type"
        assert is_llvm_floating_point(target.machine_type()), "Target type must have a floating-point machine type"

    def _name(self) -> str:
        return "FloatResize"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        def width(ty: llvmlite.ir.Type) -> int:
            if isinstance(ty, llvmlite.ir.HalfType):
                return 16
            if isinstance(ty, llvmlite.ir.FloatType):
                return 32
            if isinstance(ty, llvmlite.ir.DoubleType):
                return 64
            raise "Invalid floating point type"
        (arg, ) = args
        target, _ = self.handle_return()
        ((source, _),) = self.handle_arguments()
        source_width = width(source.machine_type())
        target_machine_type = target.machine_type()
        target_width = width(target_machine_type)
        if source_width > target_width:
            return flow.builder.fptrunc(arg, target.machine_type())
        if source_width < target_width:
            return flow.builder.fpext(arg, target.machine_type())
        return arg


class FloatToInteger(BaseTransferUnaryHandle[ControlFlow]):
    """
    Converts an integer type to a floating point type

    The source type must have a machine type of ``llvm.ir.IntType`` and the target must have a machine type of
    ``llvm.ir.HalfType``, ``llvm.ir.FloatType``, or ``llvm.ir.DoubleType``. This handle assumes the type conversion is
    acceptable for the types provided; it only cares about the machine precision.
    """
    __source: Type
    __target: Type

    def __init__(self, source: Type, target: Type, transfer: ReturnManagement = ReturnManagement.BORROW):
        super().__init__(target, source, transfer)
        assert is_llvm_floating_point(source.machine_type()), "Source is not a floating point type"
        assert isinstance(target.machine_type(), llvmlite.ir.IntType), "Target is not an integer type"

    def _name(self) -> str:
        return "FloatToInteger"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        (arg, ) = args
        return flow.builder.fptosi(arg, self.handle_return()[0].machine_type())


class ForceConversion(BaseTransferUnaryHandle[ControlFlow]):
    """
    Performs type conversion where the underlying types have the same machine type and need to conversion.

    This handle assumes the type conversion is acceptable for the types provided; it does not validate that conversion
    will yield valid results.
    """

    def __init__(self, source: Type, target: Type, transfer: ReturnManagement = ReturnManagement.BORROW):
        super().__init__(target, source, transfer)
        assert source.machine_type() == target.machine_type(), "Machine types are not compatible."

    def _name(self) -> str:
        return "NoOp"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        (arg, ) = args
        return arg


class SimpleConstant(Handle[ControlFlow]):
    """
    A handle that returns a constant value

    A handle with no arguments that returns a constant value. The constant value must be one that can be encoded in LLVM
    IR, which includes integers, floating points, arrays of the former, and strings. The exact LLVM IR produce will
    depend on the machine type.
    """
    __value: Any
    __type: Type
    __transfer: ReturnManagement

    def __init__(self, ty, value, transfer: ReturnManagement = ReturnManagement.BORROW):
        super().__init__()
        self.__type = ty
        self.__value = value
        self.__transfer = transfer

    def __str__(self) -> str:
        return f"Constant[{self.__transfer.name}]({self.__value}) → {self.__type}"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        return llvmlite.ir.Constant(self.__type.machine_type(), self.__value)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return ()

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__type, self.__transfer


class BaseExistingFunction(Handle):
    """
    A handle for an externally provided functions

    This is meant to provide a uniform base class for handling functions that can be resolved to an address to make
    a direct or indirect function call.
    """
    __return: Type
    __args: Sequence[Tuple[Type, ArgumentManagement]]
    __return_transfer: ReturnManagement

    def __init__(self,
                 return_type: Type,
                 return_transfer: ReturnManagement,
                 *args: Tuple[Type, ArgumentManagement]):
        super().__init__()
        self.__return = return_type
        self.__return_transfer = return_transfer
        self.__args = args

    def __str__(self) -> str:
        arg_str = ', '.join(f"{a}[{m}]" for a, m in self.__args)
        return f"{self._name()}[{self.__return_transfer}]({arg_str}) → {self.__return}"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        return flow.builder.call(self._function_value(flow,
                                                      self.__return.machine_type(),
                                                      [a.machine_type() for a, _ in self.__args],
                                                      args),
                                 args)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return [*self.__args]

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__return, self.__return_transfer

    def _name(self) -> str:
        """
        Gets the user-friendly name of the function as it should appear for debugging
        """
        raise NotImplementedError()

    def _function_value(self, flow: F,
                        return_type: llvmlite.ir.Type,
                        argument_types: Sequence[llvmlite.ir.Type],
                        arguments: Sequence[IRValue]) -> IRValue:
        """
        Gets an LLVM value for a function (e.g., a global function constant, or function pointer)

        This should not generate the function call; it needs only provide the function itself.

        :param flow: the control flow in which the function will be called
        :param return_type: the LLVM type of the function's return
        :param argument_types: the LLVM types of the function's arguments
        :param arguments: the actual argument values; this is provided in case a vtable lookup is required
        :return: the function value
        """
        raise NotImplementedError()


class BaseIndirectFunction(BaseExistingFunction):
    """
    A handle for calling a function from an arbitrary address provided by the Python code
    """

    def __init__(self, return_type: Type, return_transfer: ReturnManagement, *args: Tuple[Type, ArgumentManagement]):
        super().__init__(return_type, return_transfer, *args)

    def _function_value(self, flow: F, return_type: llvmlite.ir.Type, argument_types: Sequence[llvmlite.ir.Type],
                        arguments: Sequence[IRValue]) -> IRValue:
        fn_name = flow.builder.module.get_unique_name("indirect")
        fn_type = llvmlite.ir.FunctionType(return_type, argument_types)
        return flow.upsert_global_binding(fn_name, fn_type, self._address())

    def _address(self) -> ctypes.c_char_p:
        """
        The current address of the function.

        :return: the address
        """
        raise NotImplementedError()


class CurrentProcessFunction(BaseExistingFunction):
    """
    A handle from function that exists in the current running process
    """
    __name: str

    def __init__(self,
                 return_type: Type,
                 return_transfer: ReturnManagement,
                 name: str,
                 *args: Tuple[Type, ArgumentManagement]):
        super().__init__(return_type, return_transfer, *args)
        self.__name = name

    def _name(self) -> str:
        return self.__name

    def _function_value(self, flow: F, return_type: llvmlite.ir.Type,
                        argument_types: Sequence[llvmlite.ir.Type], arguments: Sequence[IRValue]) -> IRValue:
        return flow.use_native_function(self.__name, return_type, argument_types)


class LibraryFunction(BaseExistingFunction):
    """
    A handle from function that exists in an OrcJIT-loaded library
    """
    __name: str
    __tracker: ResourceTracker

    def __init__(self,
                 resource_tracker: ResourceTracker,
                 return_type: Type,
                 return_transfer: ReturnManagement,
                 name: str,
                 *args: Tuple[Type, ArgumentManagement]):
        super().__init__(return_type, return_transfer, *args)
        self.__name = name
        self.__tracker = resource_tracker

    def _name(self) -> str:
        return self.__name

    def _function_value(self, flow: F, return_type: llvmlite.ir.Type,
                        argument_types: Sequence[llvmlite.ir.Type], arguments: Sequence[IRValue]) -> IRValue:
        flow.add_library_dependency(self.__tracker.name)
        return flow.use_native_function(self.__name, return_type, argument_types)
