import ctypes
from typing import Sequence, Dict, Any

import llvmlite.ir
from llvmlite.ir import Type as LLType, IRBuilder, Value as IRValue

from dispyatcher import Type, is_llvm_floating_point, Handle, llvm_type_to_ctype, Deref
from dispyatcher.accessors import GetPointer


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


class UncheckedArray(Type, Deref, GetPointer):
    """
    A high-level type representing a C-like array.

    This array is a linear block of entries with no bounds checking or known length. It provides the same unsafe
    behaviour as C; you're welcome. It is intended mostly for compatibility with C.
    """
    __element_type: LLType

    def __init__(self, element_type: LLType):
        self.__element_type = element_type

    def __eq__(self, o: object) -> bool:
        if isinstance(o, UncheckedArray):
            return self.__element_type == o.__element_type
        else:
            return False

    def __str__(self) -> str:
        return f"Unchecked Array of {self.__element_type}"

    def target(self) -> Type:
        return MachineType(self.__element_type)

    def target_pointer(self) -> Type:
        return self

    def ctypes_type(self):
        return ctypes.POINTER(llvm_type_to_ctype(self.__element_type))

    def machine_type(self) -> LLType:
        return self.__element_type.as_pointer()


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
