import ctypes
import unittest

import llvmlite.ir

import dispyatcher
import dispyatcher.general
from dispyatcher import CallSite, Identity, IgnoreArguments
from dispyatcher.allocation import CollectIntoArray
from dispyatcher.general import MachineType, SimpleConstant
from dispyatcher.cpython import TupleUnpackingDispatcher, PY_OBJECT_TYPE, PythonControlFlowType
from dispyatcher.permute import RepeatArgument, PermuteArguments


class RepackTests(unittest.TestCase):

    def test_tuple_unpack_int(self):
        i32 = MachineType(llvmlite.ir.IntType(32))
        tud = TupleUnpackingDispatcher(IgnoreArguments(SimpleConstant(i32, -1), 0, PY_OBJECT_TYPE), 0)
        tud.append(Identity(i32), None)
        callsite = CallSite(tud, PythonControlFlowType())
        self.assertEqual(callsite(ctypes.py_object((7,))), 7)
        self.assertEqual(callsite(ctypes.py_object(7)), -1)
        self.assertEqual(callsite(ctypes.py_object((7, 7))), -1)
        self.assertEqual(callsite(ctypes.py_object("foo")), -1)
        self.assertEqual(callsite(ctypes.py_object((7.5,))), -1)

    def test_tuple_unpack_double(self):
        dbl = MachineType(llvmlite.ir.DoubleType())
        tud = TupleUnpackingDispatcher(IgnoreArguments(SimpleConstant(dbl, -1), 0, PY_OBJECT_TYPE), 0)
        tud.append(Identity(dbl), None)
        callsite = CallSite(tud, PythonControlFlowType())
        self.assertEqual(callsite(ctypes.py_object((7.4,))), 7.4)
        self.assertEqual(callsite(ctypes.py_object(7)), -1)
        self.assertEqual(callsite(ctypes.py_object((7, 7))), -1)
        self.assertEqual(callsite(ctypes.py_object("foo")), -1)
        self.assertEqual(callsite(ctypes.py_object((7,))), 7.0)


class PythonFlowTests(unittest.TestCase):

    def test_dictionary_size(self):
        handle = dispyatcher.cpython.PY_DICT_SIZE
        callsite = CallSite(handle, PythonControlFlowType())
        self.assertEqual(callsite(ctypes.py_object({})), 0)
        self.assertEqual(callsite(ctypes.py_object({"a": 1})), 1)
        self.assertEqual(callsite(ctypes.py_object({"a": 1, "b": 1})), 2)

    def test_dictionary_non_null(self):
        handle = dispyatcher.cpython.PY_DICT_GET_ITEM + dispyatcher.cpython.NullToNone + dispyatcher.Clone
        callsite = CallSite(handle, PythonControlFlowType())
        self.assertEqual(callsite(ctypes.py_object({"a": 1}), ctypes.py_object("a")), 1)
        self.assertEqual(callsite(ctypes.py_object({"a": 1}), ctypes.py_object("b")), None)

    def test_dictionary_throw_if_null(self):
        handle = dispyatcher.cpython.PY_DICT_GET_ITEM + dispyatcher.cpython.ThrowIfNull + dispyatcher.Clone
        callsite = CallSite(handle, PythonControlFlowType())
        self.assertEqual(callsite(ctypes.py_object({"a": 1}), ctypes.py_object("a")), 1)
        with self.assertRaises(ValueError):
            callsite(ctypes.py_object({"a": 1}), ctypes.py_object("b"))

    def test_value(self):
        self_handle = dispyatcher.cpython.Value(self) + dispyatcher.Clone
        callsite = CallSite(self_handle, PythonControlFlowType())
        self.assertEqual(self, callsite())

    def test_checked_cast(self):
        handle = Identity(dispyatcher.cpython.PyObjectType(object)) // dispyatcher.cpython.PyObjectType(dict)
        callsite = CallSite(handle + dispyatcher.Clone, PythonControlFlowType())
        d = {"a": 1}
        self.assertEqual(callsite(ctypes.py_object(d)), d)
        with self.assertRaises(TypeError):
            callsite(ctypes.py_object([]))

    def test_array_collection(self):
        # This is really an allocation test, but we use Python infrastructure to make it pleasant
        i8 = MachineType(llvmlite.ir.IntType(8))
        handle = CollectIntoArray(i8, 2, null_terminated=True) + dispyatcher.cpython.PY_UNICODE_FROM_STRING
        callsite = CallSite(handle, PythonControlFlowType())
        self.assertEqual(callsite(ord('h'), ord('i')), "hi")

    def test_string(self):
        callsite = dispyatcher.CallSite(dispyatcher.general.NullTerminatedString("Hi")
                                        + dispyatcher.cpython.PY_UNICODE_FROM_STRING,
                                        PythonControlFlowType())
        self.assertEqual(callsite(), "Hi")

    def test_callback(self):
        i32 = MachineType(llvmlite.ir.IntType(32))
        callback_generator = dispyatcher.cpython.callback(i32, PY_OBJECT_TYPE)
        handle = ((dispyatcher.cpython.PY_DICT_GET_ITEM_STRING << (1, dispyatcher.general.NullTerminatedString("a")))
                  + dispyatcher.cpython.ThrowIfNull
                  + callback_generator(lambda x: len(x) * 2))
        callsite = dispyatcher.CallSite(handle, PythonControlFlowType())
        self.assertEqual(callsite({"a": []}), 0)
        self.assertEqual(callsite({"a": [3]}), 2)
        self.assertEqual(callsite({"a": "blah"}), 8)

    def test_multi_unwind(self):
        handle = (dispyatcher.cpython.PY_DICT_GET_ITEM + dispyatcher.cpython.ThrowIfNull) //\
                 dispyatcher.cpython.PY_DICT_TYPE +\
                 dispyatcher.cpython.PY_DICT_GET_ITEM +\
                 dispyatcher.cpython.ThrowIfNull +\
                 dispyatcher.Clone
        callsite = CallSite(handle, PythonControlFlowType())
        self.assertEqual(callsite({"a": {"b": 1}}, "a", "b"), 1)

    def test_transient_borrow_not_capture(self):
        lookup_handle = dispyatcher.cpython.PY_DICT_GET_ITEM + dispyatcher.cpython.ThrowIfNull + dispyatcher.Clone
        callsite = CallSite(dispyatcher.cpython.TuplePackObjects(2) << lookup_handle, PythonControlFlowType())

        self.assertEqual(callsite({"a": 1}, "a", "b"), (1, "b"))

    def test_build_value(self):
        callsite = CallSite(dispyatcher.cpython.TuplePackCustom(
            dispyatcher.cpython.INT_TUPLE_ELEMENT[32],
            dispyatcher.cpython.DOUBLE_TUPLE_ELEMENT),
            PythonControlFlowType())
        self.assertEqual(callsite(3, 9.975), (3, 9.975))

    def test_permute(self):
        # This is really an allocation test, but we use Python infrastructure to make it pleasant
        i8 = MachineType(llvmlite.ir.IntType(8))
        handle = ((CollectIntoArray(i8, 4, null_terminated=True) + dispyatcher.cpython.PY_UNICODE_FROM_STRING) @
                  (PermuteArguments, RepeatArgument(0, 2, 2)))
        callsite = CallSite(handle, PythonControlFlowType())
        self.assertEqual(callsite(ord('h'), ord('i')), "hihi")
        with self.assertRaises(AssertionError):
            handle = dispyatcher.cpython.PY_OBJECT_GET_ATTR_STRING @ (PermuteArguments, RepeatArgument(0, 2))
