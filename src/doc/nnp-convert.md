nnp-convert: Convert between structure file formats
===================================================

The `nnp-convert` tool converts a given `input.data` set file (see
[format](cfg_file.md)) into a different file format. Currently, the following
file formats are supported:

 1. [Extended](http://libatoms.github.io/QUIP/io.html#module-ase.io.extxyz) XYZ,
 2. VASP POSCAR.

Usage:
------
```
nnp-convert <format> <elem1 <elem2 ...>>
            <format> ... Structure file output format (xyz/poscar).
            <elemN> .... Symbol for Nth element.
            Execute in directory with these NNP files present:
            - input.data (structure file)
```
Unfortunately, the element symbols are not automatically determined and must be
provided separated by spaces. For XYZ files the order is not important.

@warning
The order of elements given via the command line arguments is critical for
POSCAR output as the atoms will be sorted according to this list. Hence, the
order should match the one determined by the POTCAR file.

Examples:
---------

- Convert to XYZ:

  ```
  nnp-convert xyz S Cu
  ```

- Convert to POSCAR:

  ```
  nnp-convert poscar Cu S
  ```

Sample screen output:
---------------------
```
*** NNP-CONVERT ***************************************************************

Requested file format  : poscar
Output file name prefix: POSCAR
Number of elements     : 2
Element string         : Cu S
*******************************************************************************
Configuration       1:     144 atoms
Configuration       2:     144 atoms
Configuration       3:     144 atoms
Configuration       4:     144 atoms
Configuration       5:     144 atoms
Configuration       6:     144 atoms
Configuration       7:     144 atoms
Configuration       8:     144 atoms
Configuration       9:     144 atoms
Configuration      10:     144 atoms
Configuration      11:     144 atoms
Configuration      12:     144 atoms
Configuration      13:     144 atoms
Configuration      14:     144 atoms
Configuration      15:     144 atoms
Configuration      16:     144 atoms
Configuration      17:     144 atoms
Configuration      18:     144 atoms
Configuration      19:     144 atoms
Configuration      20:     144 atoms
*******************************************************************************
```

File output:
------------

- Main output depends on the chosen target file format:
  1. XYZ: `input.xyz` An XYZ file with all configurations in `input.data`.
  2. POSCAR: For each configuration a separate file with prefix `POSCAR_` is written.
- `nnp-convert.log` : Log file (copy of screen output).

