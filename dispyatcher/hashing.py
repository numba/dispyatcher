from typing import List, Sequence, Dict, Any, Iterable, Tuple, Optional

import llvmlite.ir
from llvmlite.ir import IRBuilder, Value as IRValue

from dispyatcher import ArgumentManagement, DiagramState, F, Handle, ReturnManagement, Type


class HashValueGuard:
    """
    A hash guard generates code to allow dispatching values based using hash and a table lookup.

    The ``HasValueDispatcher`` effectively writes a hash table into the generated code and the guard allows computing
    the key for an arbitrary value. Like a hash table implementation, it needs to be able to compute the hash and
    equality of a key, but, as a twist, it needs to be able to do that at compile time and emit code that does the same
    hashing at runtime.

    Since the Python ``hash`` function is unstable, it shouldn't be used in any way.

    Note that the runtime and compile time versions of this function receive separate values and the guard must figure
    out how to handle that sensibly. For instance, if hashing a string, the Python type might be ``str`` and the runtime
    type might be a C string as ``char*`` (``i8*`` in LLVM), but it could just as easily be a Python ``str`` with a
    runtime value of a Rust ``&str`` (``type{i8*, u64}`` in LLVM on a 64-bit system). In fact, the compile-time and
    runtime values can be totally decoupled. Perhaps the Python compile-time side only handles CRC32 values of strings
    while the runtime handles byte arrays.
    """

    def compatible(self, ty: Type) -> bool:
        """
        Checks if a particular type can be handled by this guard.

        :param ty: the type of the input data
        :return: true if the guard can process values of this type
        """
        pass

    def compute_hash(self, value) -> int:
        """
        Compute the hash of a particular value.

        This output must match the output of the runtime version emitted by ``generate_hash_ir``.

        :param value: the value to be hashed, as a Python value
        """
        pass

    def generate_hash_ir(self, builder: IRBuilder, arg: IRValue) -> IRValue:
        """
        Generate code to produce the hash of a value.

        :param builder: the LLVM builder
        :param arg: the value which should be hashed
        :return: the result of the hash, which must be an ``i32``
        """
        pass

    def generate_check_ir(self, value, builder: IRBuilder, arg: IRValue) -> IRValue:
        """
        Generate code to check equality of a value.

        While dispatching is done by hash, an equality check is performed after to correct any hash collisions.

        The guard is responsible for creating a correct encoding of the value in the generated code.

        The guard should do the equality comparison and provide the result.

        :param value: the value to be checked, as a Python value (the right-hand side of the comparison)
        :param builder: the LLVM builder
        :param arg: the runtime-value to be compared (the left-handle side of the comparison)
        :return: the result of the comparison (true if they are equal) as an ``i1``
        """
        pass


class HashValueDispatcher(Handle):
    """
    A handle that selects between different handles based on the inputs it receives using a hash-table

    This handle constructs a hashtable for possible code flows and then encodes the hashtable in the generated code,
    allowing fast selection of the correct code path for the input.

    Not all inputs need to be considered as part of the selection process. Each input may have a separate guard that
    selects how hashing of that input is to work. If an input should not be considered as part of the selection process
    (*e.g.*, it is an output buffer), it can have ``None`` as a guard to be excluded.

    At least one input must have a guard.
    """
    __dispatch: Dict[int, List[Tuple[Handle, Sequence[Any]]]]
    __guards: Sequence[Optional[HashValueGuard]]
    __fallback: Handle

    def __init__(self, fallback: Handle, *guards: Optional[HashValueGuard]):
        """
        Construct a new hash-dispatching handle.

        :param fallback: the handle to execute if none of the special cases match; all other handles must match the type
            of this handle
        :param guards: the guards to use; the number of guards (or `None`) must equal the number of parameters in the
            fallback handle
        """
        super().__init__()
        self.__dispatch = {}
        self.__fallback = fallback
        self.__guards = guards
        fallback.register(self)
        handle_inputs = fallback.handle_arguments()
        assert len(handle_inputs) > 0, "Cannot hash on no parameters"
        assert any(guard is not None for guard in guards), "No real guards are provided"
        assert len(guards) == len(handle_inputs), "Number of guards do not match number of parameters"
        for index, (guard, (arg_type, _)) in enumerate(zip(guards, handle_inputs)):
            assert guard.compatible(arg_type), f"Guard {guard} at index {index} is not compatible with {arg_type}"

    @property
    def fallback(self) -> Handle:
        return self.__fallback

    @fallback.setter
    def fallback(self, handle: Handle):
        assert (tuple(handle.handle_arguments()) == tuple(self.__fallback.handle_arguments()) and
                handle.handle_return() == self.__fallback.handle_return()), "Handle does not match existing signature."
        self.__fallback.unregister(self)
        handle.register(self)
        self.__fallback = handle
        self.invalidate()

    def __insert_one(self, key: Sequence[Any], target: Handle):
        self_args = self.handle_arguments()
        assert self.handle_return() == target.handle_return(),\
            f"Return type must match. Got {target.handle_return()}, expected {self.handle_return()}."
        assert tuple(self_args) == tuple(target.handle_arguments()),\
            f"Arguments must match. Got {target.handle_arguments()}, expected {self_args}."
        assert len(key) == len(self_args), "Number of arguments must match key sequence"
        value = 1
        for index, (guard, self_arg_type, target_key) in enumerate(zip(self.__guards, self_args, key)):
            if guard is not None:
                value *= guard.compute_hash(target_key)
            elif target_key is not None:
                raise ValueError(f"Key {target_key} for argument {index} is not part of the hash and should be None")

        if value not in self.__dispatch:
            self.__dispatch[value] = []
        self.__dispatch[value].append((target, key))
        target.register(self)

    def insert(self, key: Sequence[Any], target: Handle) -> None:
        """
        Adds a single new handle to the collection for this handle.

        This triggers a recompile of any call sites using this handle, so use ``extend`` or ``replace`` for doing bulk
        changes.

        :param key: a sequence of values to be fed into the guards for this handle. Any guards which are ``None`` should
            have a corresponding ``None`` in the key set. Values are read both at the time of addition and at
            recompilation, which can happen at any time; therefore, values must be immutable or effectively immutable as
            read by the guards.
        :param target: the handle to call when the arguments match the key. This handle must have the same signature as
            the fallback handle
        """
        self.__insert_one(key, target)
        self.invalidate()

    def __clear(self):
        for handles in self.__dispatch.values():
            for handle, _ in handles:
                handle.unregister(self)
        self.__dispatch.clear()

    def clear(self) -> None:
        """
        Remove all installed handles and use only the fallback handle

        This triggers a recompile of any call sites using this handle, so use ``extend`` or ``replace`` for doing bulk
        changes.
        """
        self.__clear()
        self.invalidate()

    def extend(self, collection: Iterable[Tuple[Sequence[Any], Handle]]) -> None:
        """
        Install all the key/handles provided

        See ``insert`` for details about how the key and handles are interpreted. This will trigger recompilation of any
        call sites using this handle after *all* handles have be installed.

        :param collection: the new key/handle pairs to install
        """
        for key, target in collection:
            self.__insert_one(key, target)
        self.invalidate()

    def handle_arguments(self) -> Sequence[Tuple[Type, ArgumentManagement]]:
        return self.__fallback.handle_arguments()

    def handle_return(self) -> Tuple[Type, ReturnManagement]:
        return self.__fallback.handle_return()

    def generate_handle_diagram(self, diagram: DiagramState, args: Sequence[str]) -> str:
        cases = [(repr(values), handle) for d in self.__dispatch.values() for handle, values in d]
        cases.append(("Fallback", self.__fallback))
        return diagram.dispatch(cases, args)

    def generate_handle_ir(self, flow: F, args: Sequence[IRValue]) -> IRValue:
        (self_ret_type, _) = self.__fallback.handle_return()

        value = llvmlite.ir.Constant(llvmlite.ir.IntType(32), 1)
        for index, (guard, arg_value) in enumerate(zip(self.__guards, args)):
            if guard:
                intermediate = guard.generate_hash_ir(flow.builder, arg_value)
                value = flow.builder.mul(value, intermediate)

        fallback_block = flow.builder.append_basic_block("hash_fallback")
        final_block = flow.builder.append_basic_block("hash_final")
        switch = flow.builder.switch(value, fallback_block)
        results = []
        for hash_value, handles in self.__dispatch.items():
            current_block = flow.builder.append_basic_block()
            switch.add_case(llvmlite.ir.Constant(llvmlite.ir.IntType(32), hash_value), current_block)
            flow.builder.position_at_end(current_block)
            for index, (handle, key) in enumerate(handles):
                next_block = fallback_block if index == len(handles) - 1 else flow.builder.append_basic_block(
                    f"hash_{hash_value}_idx{index}")
                for arg_idx, (key_value, arg_value, guard) in enumerate(zip(key, args, self.__guards)):
                    if guard:
                        comparison = guard.generate_check_ir(key_value, flow.builder, arg_value)
                        block = flow.builder.append_basic_block(f"hash{hash_value}_idx{index}_arg{arg_idx}")
                        flow.builder.cbranch(comparison, block, next_block)
                        flow.builder.position_at_end(block)

                result = flow.call_and_pluck(handle, [(arg, (idx,)) for idx, arg in enumerate(args)])
                results.append((result, flow.builder.basic_block))
                flow.builder.branch(final_block)
                flow.builder.position_at_end(next_block)
        flow.builder.position_at_end(fallback_block)
        result = flow.call_and_pluck(self.__fallback, [(arg, (idx,)) for idx, arg in enumerate(args)])
        results.append((result, flow.builder.basic_block))
        flow.builder.branch(final_block)

        flow.builder.position_at_end(final_block)
        phi = flow.builder.phi(self_ret_type.machine_type(), "hash_dispatch_result")
        for (value, block) in results:
            phi.add_incoming(value, block)
        return phi

    def replace(self, collection: Iterable[Tuple[Sequence[Any], Handle]]) -> None:
        """
        Remove any handles present and repopulate the dispatcher with the collection provided.

        This has the same effect as calling ``clear`` followed by ``extend``, but will ony trigger one recompilation, so
        it is preferred for bulk operations.

        :param collection: the new key/handle pairs to install
        """
        self.__clear()
        self.extend(collection)

    def __str__(self) -> str:
        dispatches = "; ".join(f"{handle} when input is {repr(key)}" for handles in self.__dispatch.values()
                               for (handle, key) in handles)
        return f"Hash value dispatch with fallback ({self.__fallback}) using [{dispatches}]"
