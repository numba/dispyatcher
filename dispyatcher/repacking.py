import ctypes
from typing import Union, List, Sequence, Iterable, Tuple, TypeVar, Generic, Dict

from llvmlite.ir import IRBuilder, Value as IRValue, Block

from dispyatcher import InvalidationTarget, Handle, Type, ControlFlow, F


class RepackingFlow(ControlFlow):
    __fallback_block: Block

    def __init__(self, ir_builder: IRBuilder, fallback_block: Block):
        super().__init__(ir_builder)
        self.__fallback_block = fallback_block

    def alternate(self):
        self.builder.branch(self.__fallback_block)

    def alternate_on_bool(self, condition: IRValue) -> None:
        ok_block = self.builder.append_basic_block("repack_ok")
        self.builder.cbranch(condition, ok_block, self.__fallback_block)
        self.builder.position_at_start(ok_block)


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

    def generate_ir(self, flow: RepackingFlow, args: Sequence[IRValue]) -> Sequence[IRValue]:
        """
        Generate code to compile this repack operation

        :param flow: the control flow to generate code in
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
        assert 0 <= common_input < len(fallback.function_type()[1]), "Common input is out of the handle argument range."
        fallback.register(self)

    @property
    def fallback(self) -> Handle:
        return self.__fallback

    @fallback.setter
    def fallback(self, handle: Handle):
        assert handle.function_type() == self.__fallback.function_type(), "Handle does not match existing signature."
        self.__fallback.unregister(self)
        handle.register(self)
        self.__fallback = handle
        self.invalidate()

    def _find_repack(self, input_args: Sequence[Type], output_args: Sequence[Type], hint: T)\
            -> Union[Repacker, None]:
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
        (self_ret_type, self_arg_types) = self.__fallback.function_type()
        (target_ret_type, target_arg_types) = target.function_type()
        assert target_ret_type == self_ret_type, "Return type must match"
        for index in range(0, self.__common):
            self_arg_type = self_arg_types[index]
            handle_arg_type = target_arg_types[index]
            assert self_arg_type == handle_arg_type,\
                f"Arguments at {index} do not match. Expected {self_arg_type} but got {handle_arg_type}."

        input_args = self_arg_types[self.__common:]
        output_args = target_arg_types[self.__common:]
        guard = self._find_repack(input_args, output_args, hint)
        assert guard is not None, f"No repacking exists from {input_args} to {output_args}"
        self_tail = self_arg_types[self.__common + guard.input_count():]
        target_tail = target_arg_types[self.__common + guard.output_count():]
        for index, (self_arg_type, handle_arg_type) in enumerate(zip(self_tail, target_tail)):
            in_idx = self.__common + guard.input_count() + index
            out_idx = self.__common + guard.output_count() + index

            assert self_arg_type == handle_arg_type, \
                f"Arguments at {in_idx} does not match {out_idx}. Expected {self_arg_type} but got {handle_arg_type}."

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

    def function_type(self) -> (Type, Sequence[Type]):
        return self.__fallback.function_type()

    def generate_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        output_block = flow.builder.append_basic_block()
        phi = []
        for index, (handle, guard) in enumerate(self.__dispatch):
            fallback_block = flow.builder.append_basic_block(f"repack_attempt_{index}")

            guard_end_index = self.__common + guard.input_count()

            inner_args = []
            inner_args.extend(args[0:self.__common])
            guard_args = args[self.__common:guard_end_index]
            inner_flow = RepackingFlow(flow.builder, fallback_block)
            inner_args.extend(guard.generate_ir(inner_flow, guard_args))
            inner_args.extend(args[guard_end_index + 1:])
            flow.extend_global_bindings(inner_flow.finish())

            result = flow.call(handle, inner_args)
            flow.builder.branch(output_block)
            phi.append((flow.builder.block, result))
            flow.builder.position_at_end(fallback_block)

        result = flow.call(self.__fallback, args)
        phi.append((flow.builder.block, result))
        flow.builder.branch(output_block)

        flow.builder.position_at_end(output_block)
        result = flow.builder.phi(self.__fallback.function_type()[0].machine_type(), "repack_result")
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
