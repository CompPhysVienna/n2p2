.. _libnnp:

The libnnp core library
=======================

.. warning::

   Documentation under construction...

The ``libnnp`` library provides all the basic ingredients for HDNNP generation or
application. For instance, it contains classes for symmetry functions, a neural
network class and data storage classes. Furthermore, a top-level class
(nnp::Mode) combines all these building blocks to form a working HDNNP setup.
Most of the provided tools and the :ref:`LAMMPS interface <if_lammps>` make use of
this Mode class and its setup methods. Consequently, the screen and log output
will often look similar. This page will walk you through the library output as
produced by the :ref:`nnp-predict` example for RPBE-D3 water (see
``examples/nnp-predict/H2O_RPBE-D3`` directory).

The initial output section (corresponding to :func:`nnp::Mode::initialize`)
is simply stating the current version, git branch and commit ID (if available).
If the library was compiled with OpenMP support, the number of used threads is
also provided (see :ref:`Parallelization <parallelization>`).

.. code-block:: none

   *******************************************************************************

      NNP LIBRARY v2.0.0
      ------------------

   Git branch  : master
   Git revision: 7b42366 (7b423664b02ff4e4979301b4a136ac3221f46be2)

   Number of OpenMP threads: 2
   *******************************************************************************

The next section (:func:`nnp::Mode::loadSettingsFile`) names which settings file is
used and how many keywords (see :ref:`Keywords <keywords>`) were found. If
problems (unknown or multiply defined keywords) occur, warnings will be issued
in this section.

.. code-block:: none

   *** SETUP: SETTINGS FILE ******************************************************

   Settings file name: input.nn
   Read 167 lines.
   Found 102 lines with keywords.
   *******************************************************************************

If data set normalization is used (see the corresponding tool :ref:`nnp-norm`
and :ref:`here <units>`) this section lists the required quantities to convert
to normalized (internal) units.

.. code-block:: none

   *** SETUP: NORMALIZATION ******************************************************

   Data set normalization is used.
   Mean energy per atom     :  -2.5521343547039809E+01
   Conversion factor energy :   2.4265748255366972E+02
   Conversion factor length :   5.8038448995319847E+00
   *******************************************************************************

.. code-block:: none

   *** SETUP: ELEMENT MAP ********************************************************

   Number of element strings found: 2
   Element  0:  H (  1)
   Element  1:  O (  8)
   *******************************************************************************

.. code-block:: none

   *** SETUP: ELEMENTS ***********************************************************

   Number of elements is consistent: 2
   Atomic energy offsets per element:
   Element  0:   0.00000000E+00
   Element  1:   0.00000000E+00
   Energy offsets are automatically subtracted from reference energies.
   *******************************************************************************

.. code-block:: none

   *** SETUP: CUTOFF FUNCTIONS ***************************************************

   Parameter alpha for inner cutoff: 0.000000
   Inner cutoff = Symmetry function cutoff * alpha
   Equal cutoff function type for all symmetry functions:
   CutoffFunction::CT_TANHU (2)
   f(r) = tanh^3(1 - r/rc)
   *******************************************************************************

.. code-block:: none

   *** SETUP: SYMMETRY FUNCTIONS *************************************************

   Abbreviations:
   --------------
   ind .... Symmetry function index.
   ec ..... Central atom element.
   ty ..... Symmetry function type.
   e1 ..... Neighbor 1 element.
   e2 ..... Neighbor 2 element.
   eta .... Gaussian width eta.
   rs ..... Shift distance of Gaussian.
   la ..... Angle prefactor lambda.
   zeta ... Angle term exponent zeta.
   rc ..... Cutoff radius.
   ct ..... Cutoff type.
   ca ..... Cutoff alpha.
   ln ..... Line number in settings file.

   Short range atomic symmetry functions element  H :
   -------------------------------------------------------------------------------
    ind ec ty e1 e2       eta        rs la zeta        rc ct   ca    ln
   -------------------------------------------------------------------------------
      1  H  2  H    1.000E-03 0.000E+00         1.200E+01  2 0.00    98
      2  H  2  O    1.000E-03 0.000E+00         1.200E+01  2 0.00   108
      3  H  2  H    1.000E-02 0.000E+00         1.200E+01  2 0.00    99
      4  H  2  O    1.000E-02 0.000E+00         1.200E+01  2 0.00   109
      5  H  2  H    3.000E-02 0.000E+00         1.200E+01  2 0.00   100
      6  H  2  O    3.000E-02 0.000E+00         1.200E+01  2 0.00   110
      7  H  2  H    6.000E-02 0.000E+00         1.200E+01  2 0.00   101
      8  H  2  O    6.000E-02 0.000E+00         1.200E+01  2 0.00   111
      9  H  2  O    1.500E-01 9.000E-01         1.200E+01  2 0.00   112
     10  H  2  H    1.500E-01 1.900E+00         1.200E+01  2 0.00   102
     11  H  2  O    3.000E-01 9.000E-01         1.200E+01  2 0.00   113
     12  H  2  H    3.000E-01 1.900E+00         1.200E+01  2 0.00   103
     13  H  2  O    6.000E-01 9.000E-01         1.200E+01  2 0.00   114
     14  H  2  H    6.000E-01 1.900E+00         1.200E+01  2 0.00   104
     15  H  2  O    1.500E+00 9.000E-01         1.200E+01  2 0.00   115
     16  H  2  H    1.500E+00 1.900E+00         1.200E+01  2 0.00   105
     17  H  3  O  O 1.000E-03 0.000E+00 -1  4.0 1.200E+01  2 0.00   162
     18  H  3  O  O 1.000E-03 0.000E+00  1  4.0 1.200E+01  2 0.00   161
     19  H  3  H  O 1.000E-02 0.000E+00 -1  4.0 1.200E+01  2 0.00   152
     20  H  3  H  O 1.000E-02 0.000E+00  1  4.0 1.200E+01  2 0.00   150
     21  H  3  H  O 3.000E-02 0.000E+00 -1  1.0 1.200E+01  2 0.00   147
     22  H  3  O  O 3.000E-02 0.000E+00 -1  1.0 1.200E+01  2 0.00   160
     23  H  3  H  O 3.000E-02 0.000E+00  1  1.0 1.200E+01  2 0.00   145
     24  H  3  O  O 3.000E-02 0.000E+00  1  1.0 1.200E+01  2 0.00   159
     25  H  3  H  O 7.000E-02 0.000E+00 -1  1.0 1.200E+01  2 0.00   142
     26  H  3  H  O 7.000E-02 0.000E+00  1  1.0 1.200E+01  2 0.00   140
     27  H  3  H  O 2.000E-01 0.000E+00  1  1.0 1.200E+01  2 0.00   137
   -------------------------------------------------------------------------------
   Short range atomic symmetry functions element  O :
   -------------------------------------------------------------------------------
    ind ec ty e1 e2       eta        rs la zeta        rc ct   ca    ln
   -------------------------------------------------------------------------------
      1  O  2  H    1.000E-03 0.000E+00         1.200E+01  2 0.00   117
      2  O  2  O    1.000E-03 0.000E+00         1.200E+01  2 0.00   127
      3  O  2  H    1.000E-02 0.000E+00         1.200E+01  2 0.00   118
      4  O  2  O    1.000E-02 0.000E+00         1.200E+01  2 0.00   128
      5  O  2  H    3.000E-02 0.000E+00         1.200E+01  2 0.00   119
      6  O  2  O    3.000E-02 0.000E+00         1.200E+01  2 0.00   129
      7  O  2  H    6.000E-02 0.000E+00         1.200E+01  2 0.00   120
      8  O  2  O    6.000E-02 0.000E+00         1.200E+01  2 0.00   130
      9  O  2  H    1.500E-01 9.000E-01         1.200E+01  2 0.00   121
     10  O  2  O    1.500E-01 4.000E+00         1.200E+01  2 0.00   131
     11  O  2  H    3.000E-01 9.000E-01         1.200E+01  2 0.00   122
     12  O  2  O    3.000E-01 4.000E+00         1.200E+01  2 0.00   132
     13  O  2  H    6.000E-01 9.000E-01         1.200E+01  2 0.00   123
     14  O  2  O    6.000E-01 4.000E+00         1.200E+01  2 0.00   133
     15  O  2  H    1.500E+00 9.000E-01         1.200E+01  2 0.00   124
     16  O  2  O    1.500E+00 4.000E+00         1.200E+01  2 0.00   134
     17  O  3  H  O 1.000E-03 0.000E+00 -1  4.0 1.200E+01  2 0.00   157
     18  O  3  O  O 1.000E-03 0.000E+00 -1  4.0 1.200E+01  2 0.00   167
     19  O  3  H  O 1.000E-03 0.000E+00  1  4.0 1.200E+01  2 0.00   156
     20  O  3  O  O 1.000E-03 0.000E+00  1  4.0 1.200E+01  2 0.00   166
     21  O  3  H  H 1.000E-02 0.000E+00 -1  4.0 1.200E+01  2 0.00   151
     22  O  3  H  H 1.000E-02 0.000E+00  1  4.0 1.200E+01  2 0.00   149
     23  O  3  H  H 3.000E-02 0.000E+00 -1  1.0 1.200E+01  2 0.00   146
     24  O  3  H  O 3.000E-02 0.000E+00 -1  1.0 1.200E+01  2 0.00   155
     25  O  3  O  O 3.000E-02 0.000E+00 -1  1.0 1.200E+01  2 0.00   165
     26  O  3  H  H 3.000E-02 0.000E+00  1  1.0 1.200E+01  2 0.00   144
     27  O  3  H  O 3.000E-02 0.000E+00  1  1.0 1.200E+01  2 0.00   154
     28  O  3  O  O 3.000E-02 0.000E+00  1  1.0 1.200E+01  2 0.00   164
     29  O  3  H  H 7.000E-02 0.000E+00 -1  1.0 1.200E+01  2 0.00   141
     30  O  3  H  H 7.000E-02 0.000E+00  1  1.0 1.200E+01  2 0.00   139
   -------------------------------------------------------------------------------
   Minimum cutoff radius for element  H: 12.000000
   Minimum cutoff radius for element  O: 12.000000
   Maximum cutoff radius (global)      : 12.000000
   *******************************************************************************

.. code-block:: none

   *** SETUP: SYMMETRY FUNCTION GROUPS *******************************************

   Abbreviations:
   --------------
   ind .... Symmetry function group index.
   ec ..... Central atom element.
   ty ..... Symmetry function type.
   e1 ..... Neighbor 1 element.
   e2 ..... Neighbor 2 element.
   eta .... Gaussian width eta.
   rs ..... Shift distance of Gaussian.
   la ..... Angle prefactor lambda.
   zeta ... Angle term exponent zeta.
   rc ..... Cutoff radius.
   ct ..... Cutoff type.
   ca ..... Cutoff alpha.
   ln ..... Line number in settings file.
   mi ..... Member index.
   sfi .... Symmetry function index.
   e ...... Recalculate exponential term.

   Short range atomic symmetry function groups element  H :
   -------------------------------------------------------------------------------
    ind ec ty e1 e2       eta        rs la zeta        rc ct   ca    ln   mi  sfi e
   -------------------------------------------------------------------------------
      1  H  2  H            *         *         1.200E+01  2 0.00     *    *    *  
      -  -  -  -    1.000E-03 0.000E+00                 -  -    -    97    1    1  
      -  -  -  -    1.000E-02 0.000E+00                 -  -    -    98    2    3  
      -  -  -  -    3.000E-02 0.000E+00                 -  -    -    99    3    5  
      -  -  -  -    6.000E-02 0.000E+00                 -  -    -   100    4    7  
      -  -  -  -    1.500E-01 1.900E+00                 -  -    -   101    5   10  
      -  -  -  -    3.000E-01 1.900E+00                 -  -    -   102    6   12  
      -  -  -  -    6.000E-01 1.900E+00                 -  -    -   103    7   14  
      -  -  -  -    1.500E+00 1.900E+00                 -  -    -   104    8   16  
      2  H  2  O            *         *         1.200E+01  2 0.00     *    *    *  
      -  -  -  -    1.000E-03 0.000E+00                 -  -    -   107    1    2  
      -  -  -  -    1.000E-02 0.000E+00                 -  -    -   108    2    4  
      -  -  -  -    3.000E-02 0.000E+00                 -  -    -   109    3    6  
      -  -  -  -    6.000E-02 0.000E+00                 -  -    -   110    4    8  
      -  -  -  -    1.500E-01 9.000E-01                 -  -    -   111    5    9  
      -  -  -  -    3.000E-01 9.000E-01                 -  -    -   112    6   11  
      -  -  -  -    6.000E-01 9.000E-01                 -  -    -   113    7   13  
      -  -  -  -    1.500E+00 9.000E-01                 -  -    -   114    8   15  
      3  H  3  H  O         *         *  *    * 1.200E+01  2 0.00     *    *    * *
      -  -  -  -  - 1.000E-02 0.000E+00 -1  4.0         -  -    -   151    1   19 1
      -  -  -  -  - 1.000E-02 0.000E+00  1  4.0         -  -    -   149    2   20 0
      -  -  -  -  - 3.000E-02 0.000E+00 -1  1.0         -  -    -   146    3   21 1
      -  -  -  -  - 3.000E-02 0.000E+00  1  1.0         -  -    -   144    4   23 0
      -  -  -  -  - 7.000E-02 0.000E+00 -1  1.0         -  -    -   141    5   25 1
      -  -  -  -  - 7.000E-02 0.000E+00  1  1.0         -  -    -   139    6   26 0
      -  -  -  -  - 2.000E-01 0.000E+00  1  1.0         -  -    -   136    7   27 1
      4  H  3  O  O         *         *  *    * 1.200E+01  2 0.00     *    *    * *
      -  -  -  -  - 1.000E-03 0.000E+00 -1  4.0         -  -    -   161    1   17 1
      -  -  -  -  - 1.000E-03 0.000E+00  1  4.0         -  -    -   160    2   18 0
      -  -  -  -  - 3.000E-02 0.000E+00 -1  1.0         -  -    -   159    3   22 1
      -  -  -  -  - 3.000E-02 0.000E+00  1  1.0         -  -    -   158    4   24 0
   -------------------------------------------------------------------------------
   Short range atomic symmetry function groups element  O :
   -------------------------------------------------------------------------------
    ind ec ty e1 e2       eta        rs la zeta        rc ct   ca    ln   mi  sfi e
   -------------------------------------------------------------------------------
      1  O  2  H            *         *         1.200E+01  2 0.00     *    *    *  
      -  -  -  -    1.000E-03 0.000E+00                 -  -    -   116    1    1  
      -  -  -  -    1.000E-02 0.000E+00                 -  -    -   117    2    3  
      -  -  -  -    3.000E-02 0.000E+00                 -  -    -   118    3    5  
      -  -  -  -    6.000E-02 0.000E+00                 -  -    -   119    4    7  
      -  -  -  -    1.500E-01 9.000E-01                 -  -    -   120    5    9  
      -  -  -  -    3.000E-01 9.000E-01                 -  -    -   121    6   11  
      -  -  -  -    6.000E-01 9.000E-01                 -  -    -   122    7   13  
      -  -  -  -    1.500E+00 9.000E-01                 -  -    -   123    8   15  
      2  O  2  O            *         *         1.200E+01  2 0.00     *    *    *  
      -  -  -  -    1.000E-03 0.000E+00                 -  -    -   126    1    2  
      -  -  -  -    1.000E-02 0.000E+00                 -  -    -   127    2    4  
      -  -  -  -    3.000E-02 0.000E+00                 -  -    -   128    3    6  
      -  -  -  -    6.000E-02 0.000E+00                 -  -    -   129    4    8  
      -  -  -  -    1.500E-01 4.000E+00                 -  -    -   130    5   10  
      -  -  -  -    3.000E-01 4.000E+00                 -  -    -   131    6   12  
      -  -  -  -    6.000E-01 4.000E+00                 -  -    -   132    7   14  
      -  -  -  -    1.500E+00 4.000E+00                 -  -    -   133    8   16  
      3  O  3  H  H         *         *  *    * 1.200E+01  2 0.00     *    *    * *
      -  -  -  -  - 1.000E-02 0.000E+00 -1  4.0         -  -    -   150    1   21 1
      -  -  -  -  - 1.000E-02 0.000E+00  1  4.0         -  -    -   148    2   22 0
      -  -  -  -  - 3.000E-02 0.000E+00 -1  1.0         -  -    -   145    3   23 1
      -  -  -  -  - 3.000E-02 0.000E+00  1  1.0         -  -    -   143    4   26 0
      -  -  -  -  - 7.000E-02 0.000E+00 -1  1.0         -  -    -   140    5   29 1
      -  -  -  -  - 7.000E-02 0.000E+00  1  1.0         -  -    -   138    6   30 0
      4  O  3  H  O         *         *  *    * 1.200E+01  2 0.00     *    *    * *
      -  -  -  -  - 1.000E-03 0.000E+00 -1  4.0         -  -    -   156    1   17 1
      -  -  -  -  - 1.000E-03 0.000E+00  1  4.0         -  -    -   155    2   19 0
      -  -  -  -  - 3.000E-02 0.000E+00 -1  1.0         -  -    -   154    3   24 1
      -  -  -  -  - 3.000E-02 0.000E+00  1  1.0         -  -    -   153    4   27 0
      5  O  3  O  O         *         *  *    * 1.200E+01  2 0.00     *    *    * *
      -  -  -  -  - 1.000E-03 0.000E+00 -1  4.0         -  -    -   166    1   18 1
      -  -  -  -  - 1.000E-03 0.000E+00  1  4.0         -  -    -   165    2   20 0
      -  -  -  -  - 3.000E-02 0.000E+00 -1  1.0         -  -    -   164    3   25 1
      -  -  -  -  - 3.000E-02 0.000E+00  1  1.0         -  -    -   163    4   28 0
   -------------------------------------------------------------------------------
   *******************************************************************************

.. code-block:: none

   *** SETUP: NEURAL NETWORKS ****************************************************

   Normalize neurons (all elements): 0
   -------------------------------------------------------------------------------
   Atomic short range NN for element  H :
   Number of weights    :   1325
   Number of biases     :     51
   Number of connections:   1376
   Architecture       27   25   25    1
   -------------------------------------------------------------------------------
      1   G   t   t   l
      2   G   t   t
      3   G   t   t
      4   G   t   t
      5   G   t   t
      6   G   t   t
      7   G   t   t
      8   G   t   t
      9   G   t   t
     10   G   t   t
     11   G   t   t
     12   G   t   t
     13   G   t   t
     14   G   t   t
     15   G   t   t
     16   G   t   t
     17   G   t   t
     18   G   t   t
     19   G   t   t
     20   G   t   t
     21   G   t   t
     22   G   t   t
     23   G   t   t
     24   G   t   t
     25   G   t   t
     26   G
     27   G
   -------------------------------------------------------------------------------
   Atomic short range NN for element  O :
   Number of weights    :   1400
   Number of biases     :     51
   Number of connections:   1451
   Architecture       30   25   25    1
   -------------------------------------------------------------------------------
      1   G   t   t   l
      2   G   t   t
      3   G   t   t
      4   G   t   t
      5   G   t   t
      6   G   t   t
      7   G   t   t
      8   G   t   t
      9   G   t   t
     10   G   t   t
     11   G   t   t
     12   G   t   t
     13   G   t   t
     14   G   t   t
     15   G   t   t
     16   G   t   t
     17   G   t   t
     18   G   t   t
     19   G   t   t
     20   G   t   t
     21   G   t   t
     22   G   t   t
     23   G   t   t
     24   G   t   t
     25   G   t   t
     26   G
     27   G
     28   G
     29   G
     30   G
   -------------------------------------------------------------------------------
   *******************************************************************************

.. code-block:: none

   *** SETUP: SYMMETRY FUNCTION SCALING ******************************************

   Equal scaling type for all symmetry functions:
   Scaling type::ST_SCALECENTER (3)
   Gs = Smin + (Smax - Smin) * (G - Gmean) / (Gmax - Gmin)
   Smin = 0.000000
   Smax = 1.000000
   Symmetry function scaling statistics from file: scaling.data
   -------------------------------------------------------------------------------

   Abbreviations:
   --------------
   ind ..... Symmetry function index.
   min ..... Minimum symmetry function value.
   max ..... Maximum symmetry function value.
   mean .... Mean symmetry function value.
   sigma ... Standard deviation of symmetry function values.
   sf ...... Scaling factor for derivatives.
   Smin .... Desired minimum scaled symmetry function value.
   Smax .... Desired maximum scaled symmetry function value.
   t ....... Scaling type.

   Scaling data for symmetry functions element  H :
   -------------------------------------------------------------------------------
    ind       min       max      mean     sigma        sf  Smin  Smax t
   -------------------------------------------------------------------------------
      1  1.09E+00  9.62E+00  2.27E+00  6.79E-01  1.17E-01  0.00  1.00 3
      2  7.33E-01  5.00E+00  1.33E+00  3.39E-01  2.34E-01  0.00  1.00 3
      3  7.60E-01  7.14E+00  1.65E+00  5.08E-01  1.57E-01  0.00  1.00 3
      4  5.48E-01  3.77E+00  1.02E+00  2.54E-01  3.11E-01  0.00  1.00 3
      5  4.01E-01  4.15E+00  9.09E-01  2.98E-01  2.67E-01  0.00  1.00 3
      6  3.62E-01  2.27E+00  6.49E-01  1.48E-01  5.25E-01  0.00  1.00 3
      7  1.89E-01  2.23E+00  4.57E-01  1.60E-01  4.90E-01  0.00  1.00 3
      8  2.67E-01  1.32E+00  4.24E-01  8.05E-02  9.49E-01  0.00  1.00 3
      9  2.45E-01  9.48E-01  3.62E-01  5.30E-02  1.42E+00  0.00  1.00 3
     10  2.22E-01  2.76E+00  5.39E-01  2.01E-01  3.94E-01  0.00  1.00 3
     11  1.47E-01  5.56E-01  2.68E-01  2.62E-02  2.45E+00  0.00  1.00 3
     12  9.91E-02  1.73E+00  2.96E-01  1.16E-01  6.14E-01  0.00  1.00 3
     13  6.51E-02  3.45E-01  1.85E-01  1.97E-02  3.57E+00  0.00  1.00 3
     14  3.17E-02  9.13E-01  1.50E-01  5.35E-02  1.13E+00  0.00  1.00 3
     15  2.92E-03  2.65E-01  7.65E-02  1.88E-02  3.82E+00  0.00  1.00 3
     16  3.21E-04  2.87E-01  4.58E-02  2.33E-02  3.49E+00  0.00  1.00 3
     17  2.47E-04  1.38E-01  1.77E-02  9.75E-03  7.23E+00  0.00  1.00 3
     18  5.10E-03  5.83E-01  2.39E-02  3.78E-02  1.73E+00  0.00  1.00 3
     19  3.23E-04  2.16E-01  1.71E-02  1.40E-02  4.63E+00  0.00  1.00 3
     20  4.96E-02  1.69E+00  1.45E-01  1.10E-01  6.11E-01  0.00  1.00 3
     21  3.41E-03  3.16E-01  1.84E-02  2.01E-02  3.20E+00  0.00  1.00 3
     22  1.31E-04  1.03E-01  6.37E-03  6.61E-03  9.76E+00  0.00  1.00 3
     23  3.38E-02  9.16E-01  8.13E-02  5.79E-02  1.13E+00  0.00  1.00 3
     24  4.17E-04  1.58E-01  4.66E-03  9.86E-03  6.35E+00  0.00  1.00 3
     25  7.35E-04  5.92E-02  3.70E-03  3.31E-03  1.71E+01  0.00  1.00 3
     26  8.98E-03  1.94E-01  2.41E-02  1.10E-02  5.40E+00  0.00  1.00 3
     27  2.12E-04  8.78E-03  2.06E-03  5.88E-04  1.17E+02  0.00  1.00 3
   -------------------------------------------------------------------------------
   Scaling data for symmetry functions element  O :
   -------------------------------------------------------------------------------
    ind       min       max      mean     sigma        sf  Smin  Smax t
   -------------------------------------------------------------------------------
      1  1.51E+00  1.00E+01  2.65E+00  6.78E-01  1.18E-01  0.00  1.00 3
      2  4.44E-01  4.62E+00  9.66E-01  3.37E-01  2.39E-01  0.00  1.00 3
      3  1.19E+00  7.53E+00  2.03E+00  5.06E-01  1.58E-01  0.00  1.00 3
      4  2.76E-01  3.39E+00  6.59E-01  2.50E-01  3.21E-01  0.00  1.00 3
      5  8.06E-01  4.54E+00  1.30E+00  2.94E-01  2.68E-01  0.00  1.00 3
      6  1.05E-01  1.89E+00  3.07E-01  1.42E-01  5.60E-01  0.00  1.00 3
      7  5.69E-01  2.62E+00  8.48E-01  1.57E-01  4.89E-01  0.00  1.00 3
      8  2.33E-02  9.36E-01  1.11E-01  6.98E-02  1.10E+00  0.00  1.00 3
      9  5.14E-01  1.85E+00  7.25E-01  9.80E-02  7.46E-01  0.00  1.00 3
     10  1.11E-01  2.91E+00  4.75E-01  2.34E-01  3.57E-01  0.00  1.00 3
     11  3.53E-01  1.07E+00  5.35E-01  4.52E-02  1.39E+00  0.00  1.00 3
     12  3.04E-02  2.53E+00  3.17E-01  2.10E-01  4.00E-01  0.00  1.00 3
     13  1.60E-01  6.63E-01  3.70E-01  3.08E-02  1.99E+00  0.00  1.00 3
     14  2.78E-03  2.30E+00  1.77E-01  1.86E-01  4.35E-01  0.00  1.00 3
     15  9.56E-03  3.91E-01  1.53E-01  2.79E-02  2.62E+00  0.00  1.00 3
     16  3.75E-06  2.04E+00  5.41E-02  1.43E-01  4.91E-01  0.00  1.00 3
     17  2.47E-03  3.43E-01  1.67E-02  2.19E-02  2.93E+00  0.00  1.00 3
     18  1.74E-05  5.63E-02  9.55E-04  3.36E-03  1.78E+01  0.00  1.00 3
     19  5.48E-02  3.02E+00  2.04E-01  2.01E-01  3.37E-01  0.00  1.00 3
     20  1.38E-03  4.99E-01  1.28E-02  3.18E-02  2.01E+00  0.00  1.00 3
     21  6.69E-03  2.67E-01  3.09E-02  1.71E-02  3.84E+00  0.00  1.00 3
     22  1.70E-02  1.42E+00  7.63E-02  9.29E-02  7.14E-01  0.00  1.00 3
     23  1.98E-02  4.08E-01  4.88E-02  2.55E-02  2.58E+00  0.00  1.00 3
     24  5.28E-04  2.33E-01  7.21E-03  1.45E-02  4.30E+00  0.00  1.00 3
     25  1.11E-05  3.53E-02  4.25E-04  2.05E-03  2.83E+01  0.00  1.00 3
     26  1.60E-02  8.22E-01  5.08E-02  5.28E-02  1.24E+00  0.00  1.00 3
     27  3.99E-03  7.86E-01  3.69E-02  5.05E-02  1.28E+00  0.00  1.00 3
     28  4.05E-05  9.84E-02  1.21E-03  5.79E-03  1.02E+01  0.00  1.00 3
     29  6.04E-03  9.93E-02  1.62E-02  5.52E-03  1.07E+01  0.00  1.00 3
     30  2.96E-03  1.55E-01  1.16E-02  8.94E-03  6.59E+00  0.00  1.00 3
   -------------------------------------------------------------------------------
   *******************************************************************************

.. code-block:: none

   *** SETUP: NEURAL NETWORK WEIGHTS *********************************************

   Weight file name format: weights.%03zu.data
   Weight file for element  H: weights.001.data
   Weight file for element  O: weights.008.data
   *******************************************************************************

.. code-block:: none

   *** SETUP: SYMMETRY FUNCTION STATISTICS ***************************************

   Equal symmetry function statistics for all elements.
   Collect min/max/mean/sigma                        : 0
   Collect extrapolation warnings                    : 0
   Write extrapolation warnings immediately to stderr: 1
   Halt on any extrapolation warning                 : 0
   *******************************************************************************
