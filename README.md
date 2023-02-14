# Dispyatch

An experimental library to implement a `MethodHandle`-like API on Python VMs.
[`MethodHandles`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/invoke/MethodHandle.html)
are a Java Virtual Machine API for dynamically filling in function calls. This
library attemps to provide a similar API accessible from Python that JITs to
Python-compatible functions to allow type conversion and dispatch operations to
be built in and consumed by Python programs.

