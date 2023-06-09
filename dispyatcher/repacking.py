from llvmlite.ir import Block, Value as IRValue
from typing import Generic, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

from dispyatcher import ArgumentManagement, F, Handle, InvalidationTarget, ReturnManagement, Type


class RepackingState:
    __flow: F
    __fallback_block: Block

    def __init__(self, flow: F, fallback_block: Block):
        self.__flow = flow
        self.__fallback_block = fallback_block

    @property
    def flow(self) -> "F":
        return self.__flow

    def alternate(self):
        self.__flow.builder.branch(self.__fallback_block)

    def alternate_on_bool(self, condition: IRValue) -> None:
        ok_block = self.__flow.builder.append_basic_block("repack_ok")
        self.__flow.builder.cbranch(condition, ok_block, self.__fallback_block)
        self.__flow.builder.position_at_start(ok_block)


class Repacker(InvalidationTarget):
    """
    A repacker is like a handle except that it can produce multiple outputs rather than a single value. This is useful
    for transforming complex objects or parsing values. It is also capable of changing flow control if failure occurs.

    This could be used to parse a string into multiple values or to unpack a tuple or list into its contents. It could
    also be used to do the reverse: collect values into a tuple or list.

    The repacker can operate over any contiguous subset of arguments.
    """

    def input_count(self) -> int:
        """
        A repacker can choose to consume some input, but does not necessarily have to consume it all.

        Any excess arguments will be matched exactly with any remaining arguments in the output.
        :return: the number of arguments the repacker intends to consume
        """
        pass

    def output_count(self) -> int:
        """
        A repacker can choose to provide some input, but does not necessarily have to provide it all.

        Any excess arguments will be matched exactly with any remaining arguments in the input.
        :return: the number of arguments the repacker intends to provide
        """

        pass

    def generate_ir(self, state: RepackingState, args: Sequence[IRValue]) -> Sequence[Tuple[IRValue, Set[int]]]:
        """
        Generate code to compile this repack operation

        :param state: the repacking wrapper around a control flow
        :param args: the subset of arguments the repacker will consume
        :return: the output arguments
        """
        pass


T = TypeVar('T')


class RepackingDispatcher(Handle, Generic[T]):
    """
    Allows "repacking" arguments to find a matching set of arguments to select a downstream path

    The primary motivation for this structure is to unpack Python tuples, which may not have a single correct type and
    choose the correct specialisation for the contents of the tuple at runtime.

    The dispatcher is initialised with a fallback handle. Additional paths maybe be provided and the dispatcher, which
    must be subclassed, can provide a repacker that allows transforming input arguments to output arguments. For
    example, suppose we have a Python tuple and we have multiple specialisations for summing the tuple, we might write:
    ```
    throw_exception_fallback = ...
    dispatcher = TupleUnpackingDispatcher(DropArguments(throw_exception, 0, py_obj), 0)
    dispatcher.append(IdentityHandle(i32), 1)
    dispatcher.append(SumInteger(i32), 2)
    ```
    Our `TupleUnpackingDispatcher` will generate repacking from a `PyObject*` to `i32` and to `i32, i32`.

    The dispatcher allows some arguments at the beginning to be the same and passed without modification. Then, the
    dispatcher can choose how many of the remaining arguments it would like to repack. Any unclaimed arguments must
    match exactly and are passed through without modification. Effectively, the dispatcher lets you specify a common
    prefix, the dispatcher then chooses the middle to change, and the rest is assumed to be a common suffix.
    """
    __dispatch: List[Tuple[Handle, Repacker]]
    __fallback: Handle
    __common: int

    def __init__(self, fallback: Handle, common_input: int):
        """
        Construct a new repacking dispatcher. This class must be subclassed to provide the `_find_repack` method.
        :param fallback: the handle to call if no input cases match. This handle defines what other handles can consider
        as input arguments
        :param common_input: the number of arguments at the beginning that are not eligible for repacking.
        """
        super().__init__()
        self.__dispatch = []
        self.__fallback = fallback
        self.__common = common_input
        assert 0 <= common_input < len(fallback.handle_arguments()), "Common input is out of the handle argument range."
        fallback.register(self)

    @property
    def fallback(self) -> Handle:
        return self.__fallback

    @fallback.setter
    def fallback(self, handle: Handle):
        assert handle.handle_return() == self.handle_return(), \
            f"Handle returns {handle.handle_return()}, but {self.handle_return()} expected."
        assert tuple(handle.handle_arguments()) == tuple(self.handle_arguments()),\
            f"Handle expects {handle.handle_arguments()}, but {self.handle_arguments()} expected."
        self.__fallback.unregister(self)
        handle.register(self)
        self.__fallback = handle
        self.invalidate()

    def _find_repack(self,
                     input_args: Sequence[Tuple[Type, ArgumentManagement]],
                     output_args: Sequence[Tuple[Type, ArgumentManagement]],
                     hint: T) -> Optional[Repacker]:
        """
        Find a possible repacking of the arguments

        Subclasses must implement this method to provide whatever logic they can use to transform input arguemtents to
        output arguments.

        :param input_args: the arguments from the input excluding the common prefix. The number of these consumed will
        be indicated by the repacker. These are a subset of the fallback handle's input arguments.
        :param output_args: the arguments from the output excluding the common prefix. The number of these provided will
        be indicated by the repacker. These are a subset of the input arguments of the handle being added.
        :param hint: way to pass implementation-specific information into the repacker
        :return: if a repacker was found, the repacker to use or `None` if there is no valid repacking
        """
        pass

    def __append(self, target: Handle, hint: T) -> None:
        self_args = self.handle_arguments()
        target_args = target.handle_arguments()
        assert target.handle_return() == self.handle_return(), \
            f"Handle returns {target.handle_return()}, but {self.handle_return()} expected."

        input_args = self_args[self.__common:]
        output_args = target_args[self.__common:]
        guard = self._find_repack(input_args, output_args, hint)
        assert guard is not None, f"No repacking exists from {input_args} to {output_args}"
        self_tail = self_args[self.__common + guard.input_count():]
        target_tail = target_args[self.__common + guard.output_count():]
        for index, (self_arg, handle_arg) in enumerate(zip(self_tail, target_tail)):
            in_idx = self.__common + guard.input_count() + index
            out_idx = self.__common + guard.output_count() + index

            assert self_arg == handle_arg, \
                f"Arguments at {in_idx} does not match {out_idx}. Expected {self_arg} but got {handle_arg}."

        self.__dispatch.append((target, guard))
        target.register(self)
        guard.register(self)

    def append(self, target: Handle, hint: T) -> None:
        """
        Adds a single handle to the dispatcher

        The handle's common prefix will be checked, then a repacker will be selected, passing the hint along, and, if
        successful, any remaining arguments will be checked as a common suffix.

        This triggers a recompile of any call sites using this handle, so use `extend` or `replace` for doing bulk
        changes.
        :param target: the handle to add
        :param hint: additional information for the dispatch process
        """
        self.__append(target, hint)
        self.invalidate()

    def __clear(self) -> None:
        for target, guard in self.__dispatch:
            target.unregister(self)
            guard.unregister(self)
        self.__dispatch.clear()

    def clear(self) -> None:
        """
        Removes all handles in the dispatch except the fallback handle

        This triggers a recompile of any call sites using this handle, so use `replace` for doing bulk changes.
        """
        self.__clear()
        self.invalidate()

    def extend(self, collection: Iterable[Tuple[Handle, T]]) -> None:
        """
        Install all the handle/hints provided

        See `append` for details about how the handles and hints are interpreted. This will trigger recompilation of any
        call sites using this handle after _all_ handles have be installed.
        :param collection: the new key/handle pairs to install
        """
        for (handle, hint) in collection:
            self.__append(handle, hint)
        self.invalidate()

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return self.__fallback.handle_arguments()

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__fallback.handle_return()

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        output_block = flow.builder.append_basic_block()
        phi = []
        for index, (handle, guard) in enumerate(self.__dispatch):
            fallback_block = flow.builder.append_basic_block(f"repack_attempt_{index}")

            guard_end_index = self.__common + guard.input_count()

            inner_args = []
            inner_args.extend((args[i], (i,)) for i in range(0, self.__common))
            guard_args = args[self.__common:guard_end_index]

            for val, indices in guard.generate_ir(RepackingState(flow, fallback_block), guard_args):
                inner_args.append((val, {i + self.__common for i in indices}))
            inner_args.extend((args[i], (i,)) for i in range(guard_end_index + 1, len(args)))

            print(args, inner_args)
            print(handle)
            result = flow.call_and_pluck(handle, inner_args)
            flow.builder.branch(output_block)
            phi.append((flow.builder.block, result))
            flow.builder.position_at_end(fallback_block)

        result = flow.call_and_pluck(self.__fallback, [(a, (i,)) for i, a in enumerate(args)])
        phi.append((flow.builder.block, result))
        flow.builder.branch(output_block)

        flow.builder.position_at_end(output_block)
        result = flow.builder.phi(self.__fallback.handle_return()[0].machine_type(), "repack_result")
        for (block, value) in phi:
            result.add_incoming(value, block)
        return result

    def replace(self, collection: Iterable[Tuple[Handle, T]]) -> None:
        """
        Remove any handles present and repopulate the dispatcher with the collection provided.

        This has the same effect as calling `clear` followed by `extend`, but will ony trigger one recompilation, so it
        is preferred for bulk operations.
        :param collection: the new handle/hint pairs to install
        """
        self.__clear()
        self.extend(collection)

    def __str__(self) -> str:
        dispatches = "; ".join(f"{handle} converting with {guard}" for (handle, guard) in self.__dispatch)
        return f"Repacking dispatch with fallback ({self.__fallback}) using [{dispatches}]"
