import itertools
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from llvmlite.ir import Value as IRValue

from dispyatcher import ArgumentManagement, DiagramState, F, Handle, PreprocessArgument, ReturnManagement,\
    TemporaryValue, Type

T = TypeVar('T')


class ArgumentPermutation:
    """
    An algorithm to re-arrange arguments
    """
    def check(self, arguments: Sequence[Tuple[Type, ArgumentManagement]]) -> Optional[str]:
        """
        Check that the arguments provided are legal under the reordering provided.

        :param arguments: the incoming arguments
        :return: ``None`` if the arguments are acceptable; otherwise an error message
        """
        pass

    def permute(self, items: Sequence[T]) -> Sequence[T]:
        """
        Rearrange the input arguments as desired by the algorithm.

        :param items: the items to rearrange
        :return: the rearranged items
        """
        pass

    def unpermute(self, items: Sequence[T]) -> Sequence[T]:
        """
        Rearrange the *output* arguments back into the input arguments.

        That is, for any input, ``x == p.permute(p.unpermute(x))```. For totally preserving permutations, this might be
        the same as ``permute``.

        :param items: the items to rearrange
        :return: the rearranged items
        """
        pass


class PermuteArguments(Handle):
    """
    Creates a new handle that permutes arguments before calling another handle.

    The permutations are user defined
    """
    __handle: Handle
    __permutations: Sequence[ArgumentPermutation]
    __is_argument_transferred: Dict[int, bool]
    __return_lifetime: List[Tuple[int, bool]]

    def __init__(self, handle: Handle, *permutations: ArgumentPermutation):
        super().__init__()
        handle.register(self)
        self.__handle = handle
        self.__permutations = permutations
        args = handle.handle_arguments()
        arg_str = ', '.join(f'{a}[{m.name}]' for a, m in args)
        for permutation in permutations:
            result = permutation.check(args)
            assert result is None,\
                f"Permutation {permutation} is not acceptable for handle with args ({arg_str}): {result}"
            args = permutation.permute(args)

    def __str__(self) -> str:
        return f"Permute[{(str(p) for p in self.__permutations)}] {self.__handle}"

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        for permutation in self.__permutations:
            args = permutation.permute(args)
        return diagram.call(self.__handle, args)

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        args = self.__handle.handle_arguments()
        for permutation in reversed(self.__permutations):
            args = permutation.unpermute(args)
        return args

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__handle.handle_return()

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> Union[TemporaryValue, IRValue]:
        call_args = [(arg, (idx,)) for idx, arg in enumerate(args)]
        for permutation in self.__permutations:
            call_args = permutation.permute(call_args)
        return flow.call(self.__handle, call_args)


class IntersperseArgument(ArgumentPermutation):
    """
    Creates interleaved duplicates of a arguments.

    Effectively turns a handle that takes *XAXBXCXD* into one that takes *XABCD* by duplicating *X*.

    The arguments must be borrowed.
    """
    __common_length: int
    __index: int
    __repeats: int
    __unique_length: int

    def __init__(self, index: int, repeats: int, common_length: int = 1, unique_length: int = 1):
        """
        Creates a repeat.

        :param index: the start of the common part of the first interleaved sequence (*i.e.*, *X*)
        :param repeats: the number of copies to create
        :param common_length: the number of common prefix arguments in each repeat (*i.e.*, the length of *X*)
        :param unique_length: the number of unique tail elements in each repeat (*i.e.*, the length of *A*)
        """
        self.__index = index
        self.__repeats = repeats
        self.__common_length = common_length
        self.__unique_length = unique_length
        assert index >= 0, "Invalid index"
        assert repeats > 0, "Invalid repeats"
        assert common_length > 0, "Invalid common length"
        assert unique_length > 0, "Invalid unique length"

    def __str__(self) -> str:
        return f"Intersperse[{self.__index}..+{self.__common_length}..+{self.__unique_length} * {self.__repeats}]"

    def check(self, arguments: Sequence[Tuple[Type, ArgumentManagement]]) -> Optional[str]:
        end_of_repeat = self.__index + (self.__common_length + self.__unique_length) * self.__repeats
        if end_of_repeat > len(arguments):
            return f"End of repeat {end_of_repeat} exceeds length {len(arguments)}"
        first_common = tuple(arguments[self.__index: self.__index + self.__common_length])
        for idx, (_, mgmt) in enumerate(first_common):
            if mgmt in (ArgumentManagement.TRANSFER_TRANSIENT, ArgumentManagement.TRANSFER_CAPTURE_PARENTS):
                return f"Repeated argument {self.__index + idx} must borrow but is {mgmt.name}"
        for repeat in range(1, self.__repeats):
            start = self.__index + (self.__common_length + self.__unique_length) * repeat
            current_repeat = tuple(arguments[start: start + self.__common_length])
            if first_common != current_repeat:
                return f"Arguments in repeat {repeat} does not match previous sequence"
        return None

    def permute(self, items: Sequence[T]) -> Sequence[T]:
        output = []
        output.extend(items[0:self.__index])
        common = items[self.__index:self.__index + self.__common_length]
        for repeat in range(self.__repeats):
            output.extend(common)
            start = self.__index + self.__common_length + repeat * self.__unique_length
            output.extend(items[start:start + self.__unique_length])
        output.extend(items[self.__index + self.__common_length + self.__repeats * self.__unique_length:])
        return output

    def unpermute(self, items: Sequence[T]) -> Sequence[T]:
        output = []
        output.extend(items[0:self.__index + self.__common_length])
        for repeat in range(self.__repeats):
            start = self.__index + (self.__common_length + self.__unique_length) * repeat + self.__common_length
            output.extend(items[start:start + self.__unique_length])
        output.extend(items[self.__index + (self.__common_length + self.__unique_length) * self.__repeats:])
        return output


class RepeatArgument(ArgumentPermutation):
    """
    Creates duplicates of an argument.

    The argument must be borrowed.
    """
    __index: int
    __repeats: int
    __length: int

    def __init__(self, index: int, repeats: int, length: int = 1):
        """
        Creates a repeat.

        :param index: the first argument to duplicate
        :param repeats: the number of copies to create
        :param length: the number of arguments to copy
        """
        self.__index = index
        self.__repeats = repeats
        self.__length = length
        assert index >= 0, "Invalid index"
        assert repeats > 0, "Invalid repeats"
        assert length > 0, "Invalid length"

    def __str__(self) -> str:
        return f"Repeat[{self.__index}..+{self.__length} * {self.__repeats}]"

    def check(self, arguments: Sequence[Tuple[Type, ArgumentManagement]]) -> Optional[str]:
        end_of_repeat = self.__index + self.__length * self.__repeats
        if end_of_repeat > len(arguments):
            return f"End of repeat {end_of_repeat} exceeds length {len(arguments)}"
        first_repeat = tuple(arguments[self.__index: self.__index + self.__length])
        for idx, (_, mgmt) in enumerate(first_repeat):
            if mgmt in (ArgumentManagement.TRANSFER_TRANSIENT, ArgumentManagement.TRANSFER_CAPTURE_PARENTS):
                return f"Repeated argument {self.__index + idx} must borrow but is {mgmt.name}"
        for repeat in range(1, self.__repeats):
            start = self.__index + self.__length * repeat
            current_repeat = tuple(arguments[start: start + self.__length])
            if first_repeat != current_repeat:
                return f"Arguments in repeat {repeat} does not match previous sequence"
        return None

    def permute(self, items: Sequence[T]) -> Sequence[T]:
        output = []
        output.extend(items[0:self.__index])
        output.extend(itertools.chain.from_iterable(itertools.repeat(
            items[self.__index: self.__index + self.__length], self.__repeats)))
        output.extend(items[self.__index + self.__length:])
        return output

    def unpermute(self, items: Sequence[T]) -> Sequence[T]:
        output = []
        output.extend(items[0:self.__index + self.__length])
        output.extend(items[self.__index + self.__length * self.__repeats:])
        return output


class ReverseArguments(ArgumentPermutation):
    """
    Reverses all the arguments
    """

    def check(self, arguments: Sequence[Tuple[Type, ArgumentManagement]]) -> Optional[str]:
        return True

    def permute(self, items: Sequence[T]) -> Sequence[T]:
        return list(reversed(items))

    def unpermute(self, items: Sequence[T]) -> Sequence[T]:
        return self.permute(items)

    def __str__(self) -> str:
        return "Reverse"


class SwapArguments(ArgumentPermutation):
    """
    Swaps two non-overlapping blocks of arguments of the same length
    """
    __source: int
    __destination: int
    __length: int

    def __init__(self, source: int, destination: int, length: int = 1):
        """
        Create a new swap.

        :param source: the start index for swapping
        :param destination:  the target index of swapping
        :param length: the number of arguments to swap
        """
        assert source >= 0
        assert destination >= 0
        assert length > 0
        assert min(source, destination) + length < max(source, destination), "Cannot swap overlapping ranges"
        self.__source = source
        self.__destination = destination
        self.__length = length

    def check(self, arguments: Sequence[Tuple[Type, ArgumentManagement]]) -> Optional[str]:
        for start, name in ((self.__source, "Source"), (self.__destination, "Destination")):
            if start + self.__length > len(arguments):
                return f"{name} range [{start}:{start + self.__length}] exceeds length {len(arguments)}"
        return None

    def permute(self, items: Sequence[T]) -> Sequence[T]:
        output = []
        output.extend(items[0:self.__source])
        output.extend(items[self.__destination:self.__destination + self.__length])
        output.extend(items[self.__source + self.__length:self.__destination])
        output.extend(items[self.__source:self.__source + self.__length])
        output.extend(items[self.__destination + self.__length:])
        return output

    def unpermute(self, items: Sequence[T]) -> Sequence[T]:
        return self.permute(items)

    def __str__(self) -> str:
        src_end = self.__source + self.__length
        dest_end = self.__destination + self.__length
        return f"Swap [{self.__source}:{src_end}] and [{self.__destination}:{dest_end}]"


def implode_args(handle: Handle, index: int, *preprocessors: Handle) -> Handle:
    """
    Takes a handle and preprocess arguments duplicating input arguments.

    This is intended to be a convenient way to unpack a structure into individual arguments.

    Given a handle ``t f(t0, t1, t2, t3)`` and preprocessors ``t1 a(x)``, ``t2 b(x)``, calling
    ``implode_args(f, 1, a, b)``, will create a handle ``t i(t0, x, t3)`` equivalent to
    ``i(v0, v1, v2) = f(v0, a(v1), b(v1), v2)``.

    :param handle: the handle to transform
    :param index: the start position in the argument to handle to replace
    :param preprocessors: the number of arguments to replace with the output of the provided handles. The handles must
        all have the same arguments
    :return: the preprocessed handle
    """
    if len(preprocessors) == 0:
        return handle
    handle_args = handle.handle_arguments()
    assert 0 <= index < len(handle_args), "Index is out of range"
    assert index + len(preprocessors) < len(handle_args), "Too many preprocessor for handle"
    required_prep_args = None
    for prep_idx, preprocessor in enumerate(preprocessors):
        prep_args = preprocessor.handle_arguments()
        if required_prep_args is None:
            required_prep_args = prep_args
        else:
            assert prep_args == required_prep_args,\
                f"Arguments {prep_args} for preprocessor {prep_idx} do not match previous {required_prep_args}"
        handle = PreprocessArgument(handle, index + prep_idx * len(required_prep_args), preprocessor)
    return PermuteArguments(handle, RepeatArgument(index, len(preprocessors), len(required_prep_args)))
