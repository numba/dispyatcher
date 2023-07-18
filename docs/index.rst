JVM ``MethodHandles`` in Python
===============================

Yes, Java's most useful and inscrutable technology is now available to you, in Python, using LLVM.

A handle represents a small bit of executable code defined only by its input and output. Handles can be composed into larger executable bits of code and put into a callsite. The callsite can then be used just like a Python function. The callsite can be updated to have different code inside of it, as long as the function signature is the same.

Why is this desirable? The JVM uses this technology to allow defining incomplete programs that can be expanded as they run. This is particularly helpful for being able to generate dispatch tables that didn't match the JVM's original design (*e.g.*, multiple dispatch is not a feature supported in the JVM, but it can be built using ``MethodHandles``) or for dynamically compiling specializations of a function.

This implementation varies from the JVM implementation because the JVM version provides certain features, including garbage collection and exception handling, that aren't available in C and this library is meant to allow interaction with CPython and any sufficiently well-behaved C code.

Each handle wraps a small well-defined piece of machine code. Generally, handles are meant to be very simple primitive operations that can be composed together, so most handles are only a few lines of machine code. One of the core handles is just to call a function and each handle, like a function, has a signature. The signature defines:

* the return type
* the return management
* the argument types and management
* the control flow type

Let's start with types, as they are most familiar. Dispyatcher doesn't impose a type system on you. Instead, it requires that you capture the semantics of your type system. Out of the box, it provides two type systems: ``MachineType``, which closely matches C and LLVM's type system, and ``PyObjectType``, that models CPython's type system. Dispyatcher has pretty minimal requirements on your type system. It needs:

* to know the LLVM type for a type, so it can generate appropriate data structures
* conversion rules between types
* to generate code to manage memory with a clone/copy/increment reference-count-function and free/drop/decrement-reference-count function.

Explicit copying is required even for primitive types, which is tedious. There's no conception of Rust's ``Clone`` vs ``Copy`` semantics.

Handles can be cast using the conversion rules provided by the type system using the division operators. ``handle // ty`` will adapt the handle to one that returns ``ty``. ``handle / (t1, t2, t3)``, will cast the arguments, individually, to a handle that takes ``(t1, t2, t3)`` as arguments.

Next, let's talk about management. In the JVM, the garbage collector manages everything, so any intermediate objects are collected by the JVM. C and CPython require explicit management of memory. That is extremely tedious and interferes with making composition work easily. To get around this, dispyatcher tries something like an automatic version of Rust's lifetimes. Every handle must produce output and this output will either be *borrowed*, callee owns the memory, or *transferred*, caller owns the memory. With this information, dispyatcher can determine if it needs to destroy the value. Arguments also have the same semantics of being *borrowed*, caller own the memory, or *transferred*, callee owns the memory. The lifetimes comes in how return values are connected to argument values. Suppose we had a handle that adds two integers; the output value does not depend on the input values remaining live. This is *transient* access, since the handle may read those values, but the output doesn't track them. A handle that gets an item from an array will transiently use the index, but the return value will *capture* the array. That is, it is not safe to free a value that has been captured until those dependent values are first freed. Sometimes, values are indirectly capture. For instance, if there is an array and an iterator is created, the iterator captures the array, but if you read the current value of the iterator, that value is in the array, not the iterator, so it *captures parents*. Rust allows very fine grained selection of lifetimes, but we're much less subtle here.

Finally, let's talk control flow. Unlike a general purpose programming language, handles don't support complicated flow control. However, there are lots of cases where things can go wrong, so handles need an exceptional program flow mechanism. There is a default flow control that is infallible (*i.e.*, it supports no exceptions). There is a Python flow control that uses CPython's API to create Python exceptions. It is possible to build other flow control mechanisms that could interact with other exception management APIs.

So, the dispyatcher API provide handles to wrap externally written functions and a number of small primitive operations (*e.g.*, dereferencing pointers, accessing arrays, accessing structures).

Handles can be composed using the ``+`` operator. This assumes the second handle can take the first handle's output as input. 

Some of these operations are generic over the type they support. For instance, ``Clone`` creates a copy of a value and works over any type. These can be used directly in a ``+`` operation and will be called with the first argument as the return type of the left hand argument. That is: ``Identity(ty) + Clone`` is short-hand for ``Identity(ty) + Clone(ty)``. Sometimes additional information is require, so a tuple can be used: ``handle + (GetElementPointer, 3)``

Rarely, operations need to swallow a whole other handle. For instance, the ``WithoutGIL`` operation wraps another handle and ensures everything that handle executes happens without the Python GIL being held. The ``@`` operator can be used for that: ``handle @ WithoutGIL`` is short of ``WithoutGIL(handle)``.

These operators are provided to allow something that looks like a chain of wrapping operations.

Once a handle is constructed, creating a ``Callsite`` will compile it and turn it into a callable object. The control flow can choose the exact semantics for calling the handle. The infallible control flow uses a standard Python ``CFUNC`` while the Python control flow uses ``PYFUNC``.

.. automodule:: dispyatcher
    :members:

.. automodule:: dispyatcher.accessors
    :members:

.. automodule:: dispyatcher.cpython
    :members:

.. automodule:: dispyatcher.general
    :members:

.. automodule:: dispyatcher.hashing
    :members:

.. automodule:: dispyatcher.permute
    :members:

.. automodule:: dispyatcher.repacking
    :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
