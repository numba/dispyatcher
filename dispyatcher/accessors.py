from typing import Optional, Sequence, Tuple
from llvmlite.ir import ArrayType, BaseStructType, IntType, PointerType, Value as IRValue
from dispyatcher import ArgumentManagement, ControlFlow, F, Handle, Type, ReturnManagement


class GetMember(Type):
    """
    A type which has the layout of a struct (a collection of heterogeneously typed values)
    """

    def member_target(self, member: int) -> Type:
        """
        Get the type of a member by index
        :param member: the member index
        :return: the type of that member
        """
        pass


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
        Gets the type of an element in an array-like structure
        :return: the element type
        """
        pass


class ArrayElementPointer(Handle[ControlFlow]):
    """
    A handle that takes an array and an index access the element at the provided index.

    Unlike `GetElementPointer`, this operates on a dynamic index. No bounds checking is provided.
    """
    __container_type: Type
    __index_type: Type
    __element_type: Type

    def __init__(self, container_type: Type, index_type: Type, element_type: Optional[Type] = None):
        """
        Creates a new array lookup handle
        :param container_type: the type of the array; it must be an LLVM array or pointer type
        :param index_type: the type of the index, which must have an LLVM type that is a 32-bit or 64-bit integer
        :param element_type: the type of the element being pointed to; if none, the container type must implement
        `GetPointer`
        """
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

    def __str__(self) -> str:
        return f"ArrayElementPointer({self.__container_type}, {self.__index_type}) → {self.__element_type}"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        return flow.builder.gep(args[0], args[1])

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return ((self.__container_type, ArgumentManagement.BORROW_CAPTURE),
                (self.__index_type, ArgumentManagement.BORROW_TRANSIENT))

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__element_type, ReturnManagement.BORROW


class GetElementPointer(Handle):
    """
    Get the address of the element in a heterogenous or homogenous container type

    This is the equivalent of C's `&x->foo` or `&x[3]`.
    """
    __index: int
    __container_type: Type
    __field_type: Type

    def __init__(self, container_type: Type, index: int, field_ty: Optional[Type] = None):
        """
        Create a handle that can access the contents of a container type
        :param container_type: the type of the container, which must have an LLVM type of struct, pointer, or array
        :param index: the element to access
        :param field_ty: the type of the element being referenced. If none, the container type needs to implement
        `GetPointer` or `GetMember` to return the type of the element
        """
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
            assert container_llvm_type.elements, "Structure does not have elements defined"
            assert 0 <= index < len(container_llvm_type.elements),\
                f"Element index {index} is not in range {len(container_llvm_type.elements)}"
            inner_type = container_llvm_type.elements[index].as_pointer()
            if field_ty is None and isinstance(container_type, GetMember):
                field_ty = container_type.member_target(index)
        else:
            raise TypeError(f"Type {container_llvm_type} is not a container")

        if field_ty is None:
            raise TypeError("Field type is not provided and cannot be inferred")
        assert inner_type == field_ty.machine_type(),\
            f"Field type {inner_type} does not match expected type {field_ty.machine_type()}"

    def __str__(self) -> str:
        return f"GetElementPointer[{self.__index}]({self.__container_type}) → {self.__field_type}"

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        return flow.builder.gep(args[0], IntType(32)(self.__index))

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return (self.__container_type, ArgumentManagement.BORROW_CAPTURE),

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__field_type, ReturnManagement.BORROW
