import ctypes
import unittest

import llvmlite.ir

import dispyatcher
import dispyatcher.general
from dispyatcher import CallSite, Identity, IgnoreArguments
from dispyatcher.allocation import CollectIntoArray
from dispyatcher.general import MachineType, SimpleConstant
from dispyatcher.cpython import TupleUnpackingDispatcher, PY_OBJECT_TYPE, PythonControlFlowType


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
