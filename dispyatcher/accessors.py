import ctypes
from typing import Dict, Sequence, Union
from dispyatcher import Handle, Type
from llvmlite.ir import ArrayType, BaseStructType, PointerType, IntType, IRBuilder, Value as IRValue


class GetPointer(Type):
    """
    Allows accessing array-like structures

    This allows creation of handles that use LLVM's `get-element-pointer` to index into a structure and know the type
    of the element in that structure. In C-terms, it answers `typeof &x[0]`. This might seem redundant since, for C
    arrays, this will always be the original type. However, it may be desirable to create an opaque array where the type
    of the elements is different from the type of the array. A reason to do this is to avoid making `&(&x[1])[1]`
    possible. This can also be implemented separately from deref, so `&x[0]` is possible, but `*x` is not.

    This may or may not be applicable to every situation, but it designed to allow creating more sophisticated type
    checking using handles than LLVM or C would allow.
    """

    def target_pointer(self) -> Type:
        """
        Gets type of an element in an array-like structure
        :return: the element type
        """
        pass


class GetMember(Type):

    def member_target(self, member: int) -> Type:
        pass


class GetElementPointer(Handle):
    __index: int
    __container_type: Type
    __field_type: Type

    def __init__(self, container_type: Type, index: int, field_ty: Union[Type, None] = None):
        super().__init__()
        self.__index = index
        self.__container_type = container_type
        self.__field_type = field_ty
        container_llvm_type = container_type.machine_type()
        inner_type: Type

        if isinstance(container_llvm_type, ArrayType):
            inner_type = container_llvm_type.element.as_pointer()
            assert 0 <= index < container_llvm_type.count,\
                f"Element index {index} is not in range {container_llvm_type.count}"
            if field_ty is None and isinstance(container_type, GetPointer):
                field_ty = container_type.target_pointer()
        elif isinstance(container_llvm_type, PointerType):
            inner_type = container_llvm_type.pointee.as_pointer()
            if field_ty is None and isinstance(container_type, GetPointer):
                field_ty = container_type.target_pointer()
        elif isinstance(container_llvm_type, BaseStructType):
            assert container_type.elements, "Structure does not have elements defined"
            assert 0 <= index < len(container_llvm_type.element),\
                f"Element index {index} is not in range {len(container_llvm_type.elements)}"
            inner_type = container_llvm_type.elements[index].as_pointer()
            if field_ty is None and isinstance(container_type, GetMember):
                field_ty = container_type.member_target(index)
        else:
            raise TypeError(f"Type {container_llvm_type} is not a container")

        if field_ty is None:
            raise TypeError("Field type is not provided and cannot be inferred")
        assert inner_type.machine_type() == field_ty.machine_type(),\
            f"Field type {inner_type} does not match expected type {field_ty.machine_type()}"

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        return builder.gep(args[0], IntType(32)(self.__index))

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__field_type, (self.__container_type,)

    def __str__(self) -> str:
        return f"Element {self.__index} {self.__field_type} of {self.__container_type}"


class ArrayElementPointer(Handle):
    __container_type: Type
    __index_type: Type
    __element_type: Type

    def __init__(self, container_type: Type, index_type: Type, element_type: Union[Type, None] = None):
        super().__init__()
        self.__container_type = container_type
        self.__index_type = index_type
        assert isinstance(index_type.machine_type(), IntType), f"{index_type} is not an integer type"
        assert index_type.machine_type().width in (32, 64), f"{index_type} is not a valid pointer size"
        container_llvm_type = container_type.machine_type()
        assert isinstance(container_llvm_type, (ArrayType, PointerType)),\
            f"Type {container_llvm_type} is not a container"
        if element_type is None:
            if isinstance(container_type, GetPointer):
                self.__target_type = container_type.target_pointer()
            else:
                raise TypeError("Container type does not implement get-pointer and no target type is not provided")
        else:
            self.__element_type = element_type
            target_type = container_llvm_type.element.as_pointer()
            assert target_type == element_type.machine_type(),\
                f"Array elements {target_type} do not match expected type {element_type.machine_type()}"

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        return builder.gep(args[0], args[1])

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__element_type, (self.__container_type, self.__index_type)

    def __str__(self) -> str:
        return f"Array element pointer {self.__container_type} with {self.__index_type} yielding {self.__element_type}"


