LAMMPS NNP interface
====================

Purpose
-------

The LAMMPS interface adds the neural network potential method in LAMMPS. In this
way the ability of a previously fitted NN to predict energies and forces is
combined with the usability and features of a large MD simulation package. It
fully supports parallelization via MPI.

Build instructions
------------------

The LAMMPS interface has been tested with version `11Aug17`. To build LAMMPS
(residing in a directory called `<LAMMPS>`) with support for neural network
potentials follow these steps: First, build the required libraries:
```
cd <n2p2>/src
make libnnpif-shared
```
For static linking use the target `libnnpif-static` instead. Then change to the
LAMMPS root directory and link the `<n2p2>` folder to the `lib` subdirectory
(with name `nnp`):
```
cd <LAMMPS>/
ln -s <n2p2> lib/nnp
```
Next, copy the USER-NNP package to the LAMMPS source directory:
```
cp -r <n2p2>/src/interface/LAMMPS/src/USER-NNP <LAMMPS>/src
```
Finally activate the NNP package in LAMMPS:
```
cd <LAMMPS>/src
make yes-user-nnp
```
Now, you can compile LAMMPS for your target as usual:
```
make <target>
```
_Note_: If you want to compile a serial version of LAMMPS with neural network
potential support, the use of MPI needs to be deactivated for `libnnpif`. Before
the build process enable the `-DNOMPI` option and select a non-MPI compiler in
the corresponding makefile (`<n2p2>/src/libnnpif/makefile`).

_Note_: If dynamic linking (`libnnpif-shared`) is used, you need to make the NNP
libraries visibile in your system, e.g. add this line in your `.bashrc`:
```
export LD_LIBRARY_PATH=<n2p2>/lib:${LD_LIBRARY_PATH}
```

Usage
-----

The neural network potential method is introduced in the context of a pair style
named `nnp`. LAMMPS comes with a large collection of these pair styles, e.g. for
a LJ or Tersoff potential, look
[here](http://lammps.sandia.gov/doc/pair_style.html) for more information. The
setup of a `nnp` pair style is done by issuing two commands: `pair_style` and
`pair_coeff`. See [this page](../../src/doc/pair_nnp.html) for a detailed
description.
