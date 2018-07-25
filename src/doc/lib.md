NNP libraries
=============

Build instructions
------------------

There are three libraries provided in the NNP repository which allow to combine
different functionality related to neural network potentials:

- `libnnp`
- `libnnpif`
- `libnnptrain`

All libraries share the same build instructions: Just switch to the
corresponding directory and type:
```
make
```
This will compile both a shared and a static version of the library and put
binary files to `<n2p2>/lib` and headers to `<n2p2>/include` directories. If only
one version is needed type
```
make shared
```
or
```
make static
```
instead. The `libnnpif` and `libnnptrain` libraries require headers from
`libnnp` and can only be used in conjunction with the core library. Thus, we
always need to build `libnnp` in advance.

`libnnp`
--------

`libnnpif`
----------

`libnnptrain`
-------------

