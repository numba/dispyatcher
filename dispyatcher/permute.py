import ctypes
import itertools
from typing import Dict, Sequence, TypeVar
from dispyatcher import Handle, Type, PreprocessArgumentHandle
from llvmlite.ir import IRBuilder, Value as IRValue


T = TypeVar('T')


class ArgumentPermutation:
    def check(self, arguments: Sequence[Type]) -> bool:
        pass

    def permute(self, items: Sequence[T]) -> Sequence[T]:
        pass


class PermuteArgumentsHandle(Handle):
    """
    Creates a new handle that permutes arguments before calling another handle.

    The permutations are user defined
    """
    __handle: Handle
    __permutations: Sequence[ArgumentPermutation]

    def __init__(self, handle: Handle, *permutations: ArgumentPermutation):
        super().__init__()
        handle.register(self)
        self.__handle = handle
        self.__permutations = permutations
        (_, args) = handle.function_type()
        for permutation in permutations:
            assert permutation.check(args), f"Permutation {permutation} is not acceptable for handle with args {args}"

    def function_type(self) -> (Type, Sequence[Type]):
        (handle_ret, handle_args) = self.__handle.function_type()
        for permutation in self.__permutations:
            handle_args = permutation.permute(handle_args)

        return handle_ret, handle_args

    def generate_ir(self, builder: IRBuilder, args: Sequence[IRValue],
                    global_addresses: Dict[str, ctypes.c_char_p]) -> IRValue:
        for permutation in self.__permutations:
            args = permutation.permute(args)
        return self.__handle.generate_ir(builder, args, global_addresses)

    def __str__(self) -> str:
        return f"Permute arguments of {self.__handle} using {(str(p) for p in self.__permutations)}"


class CopyArgument(ArgumentPermutation):
    __index: int
    __count: int
    __chunk: int

    def __init__(self, index: int, count: int, chunk: int = 1):
        self.__index = index
        self.__count = count
        self.__chunk = chunk
        assert index >= 0, "Invalid index"
        assert count > 0, "Invalid count"
        assert chunk > 0, "Invalid chunk size"

    def check(self, arguments: Sequence[Type]) -> bool:
        return self.__index + self.__chunk < len(arguments)

    def permute(self, items: Sequence[T]) -> Sequence[T]:
        output = []
        output.extend(items[0:self.__index])
        output.extend(itertools.chain.from_iterable(itertools.repeat(
            items[self.__index: self.__index + self.__chunk], self.__count)))
        output.extend(items[self.__index:])
        return output


class SwapArguments(ArgumentPermutation):
    __source: int
    __destination: int
    __length: int

    def __init__(self, source: int, destination: int, length: int):
        assert source >= 0
        assert destination >= 0
        assert length > 0
        self.__source = source
        self.__destination = destination
        self.__length = length

    def check(self, arguments: Sequence[Type]) -> bool:
        return self.__source + self.__length < len(arguments) and self.__destination + self.__length < len(arguments)

    def permute(self, items: Sequence[T]) -> Sequence[T]:
        output = []
        output.extend(items[0:self.__source])
        output.extend(items[self.__destination:self.__destination + self.__length])
        output.extend(items[self.__source + self.__length:self.__destination])
        output.extend(items[self.__source:self.__source + self.__length])
        output.extend(items[self.__destination + self.__length:])
        return output

    def __str__(self) -> str:
        src_end = self.__source + self.__length
        dest_end = self.__destination + self.__length
        return f"Swap [{self.__source}:{src_end}] and [{self.__destination}:{dest_end}]"


class ReverseArguments(ArgumentPermutation):

    def check(self, arguments: Sequence[Type]) -> bool:
        return True

    def permute(self, items: Sequence[T]) -> Sequence[T]:
        return list(reversed(items))

    def __str__(self) -> str:
        return "Reverse"


def implode_args(handle: Handle, index: int, *preprocessors: Handle) -> Handle:
    """
    Takes a handle and preprocess arguments duplicating a input arguments.

    This is intended to be a convenient way to unpack a structure into individual arguments.

    Given a handle `t f(t0, t1, t2, t3)` and preprocessors `t1 a(x)`, `t2 b(x)`, calling `implode_args(f, 1, a, b)`,
    will create a handle `t i(t0, x, t3)` equivalent to `I(v0, v1, v2) = f(v0, a(v1), b(v1), v2)`
    :param handle: the handle to transform
    :param index: the start position in the argument to handle to replace
    :param preprocessors: the number of arguments to replace with the output of the provided handles. The handles must
    all have the same arguments
    :return: the preprocessed handle
    """
    (handle_ret, handle_args) = handle.function_type()
    assert 0 <= index < len(handle_args), "Index is out of range"
    assert index + len(preprocessors), "Too many preprocessor for handle"
    if len(preprocessors) == 0:
        return handle
    required_prep_args = None
    for prep_idx, preprocessor in enumerate(preprocessors):
        (prep_ret, prep_args) = preprocessor.function_type()
        if required_prep_args is None:
            required_prep_args = prep_args
        else:
            assert prep_args == required_prep_args,\
                f"Arguments {prep_args} for preprocessor {prep_idx} do not match previous {required_prep_args}"
        required_arg = handle_args[index + prep_idx]
        assert prep_ret == required_arg, f"Preprocessor {prep_idx} emits {prep_ret}, but {required_arg} expected"
        handle = PreprocessArgumentHandle(handle, index + prep_idx, preprocessor)
    return PermuteArgumentsHandle(handle, CopyArgument(index, len(preprocessors), len(required_prep_args)))

