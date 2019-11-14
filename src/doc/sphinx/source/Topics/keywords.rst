.. _keywords:

NNP configuration: keywords
===========================

.. warning::

   Documentation under construction...

The NNP settings file (usually named ``input.nn``\ ) contains the setup of neural
networks and symmetry functions. Each line may contain a single keyword with no,
one or multiple arguments. Keywords and arguments are separated by at least one
whitespace. Lines or part of lines can be commented out with the symbol "#",
everything right of "#" will be ignored. The order of keywords is not
important, most keywords may only appear once (exceptions: ``symfunction_short``
and ``atom_energy``\ ). Here is the list of available keywords (if no usage
information is provided, then the keyword does not support any arguments).

General NNP keywords
--------------------

These keywords provide the basic setup for a neural network potential.
Therefore, they are mandatory for most NNP applications and a minimal example
setup file would contain at least:

.. code-block:: none

   number_of_elements 1
   elements H
   global_hidden_layers_short 2
   global_nodes_short 10 10
   global_activation_short t t l
   cutoff_type 1
   symfunction_short ...
   ...

These commands will set up a neural network potential for hydrogen atoms (2
hidden layers with 10 neurons each, hyperbolic tangent activation function) and
the cosine cutoff function for all symmetry functions. Of course, further
symmetry functions need to be specified by adding multiple lines starting with
the ``symfunction_short`` keyword.

----

``number_of_elements``
^^^^^^^^^^^^^^^^^^^^^^

Defines the number of elements the neural network potential is designed for.

**Usage:**
   ``number_of_elements <integer>``

**Examples**:
   ``number_of_elements 3``

----

``elements``
^^^^^^^^^^^^

**Usage**:
   ``elements <string(s)>``

**Examples**:
   ``elements O H Zn``

This keyword defines all elements via a list of element symbols. The number
of items provided has to be consistent with the argument of the
``number_of_elements`` keyword. The order of the items is not important,
elements are automatically sorted according to their atomic number. The list
:member:`nnp::ElementMap::knownElements` contains a list of recognized element
symbols.

----

``atom_energy``
^^^^^^^^^^^^^^^

**Usage**:
   ``atom_energy <string> <float>``

**Examples**:
   ``atom_energy O -74.94518524``

Definition of atomic reference energy. Shifts the total potential energy by
the given value for each atom of the specified type. The first argument is the
element symbol, the second parameter is the shift energy.

----

``global_hidden_layers_short``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Usage**:
   ``global_hidden_layers_short <integer>``

**Examples**:
   ``global_hidden_layers_short 3``

Sets the number of hidden layers for neural networks of all elements.

----

``global_nodes_short``
^^^^^^^^^^^^^^^^^^^^^^

**Usage**:
   ``global_nodes_short <integer(s)>``

**Examples**:
   ``global_nodes_short 15 10 5``

Sets the number of neurons in the hidden layers of neural networks of all
elements. The number of integer arguments has to be consistent with the
argument of the ``global_hidden_layers_short`` keyword. Note: The number of
input layer neurons is determined by the number of symmetry functions and there
is always only a single output neuron.

----

``global_activation_short``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Usage**:
   ``global_activation_short <chars>``

**Examples**:
   ``global_activation_short t t t l``

Sets the activation function per layer for hidden layers and output layers
of neural networks of all elements. The number of integer arguments has to
be consistent with the argument of the ``global_hidden_layers_short`` keyword
(i.e. number of hidden layers + 1). Activation functions are chosen via
single characters from the following table (see also
:type:`nnp::NeuralNetwork::ActivationFunction`).

.. list-table::
   :header-rows: 1

   * - Character
     - Activation function type
   * - l
     - :enumerator:`nnp::NeuralNetwork::AF_IDENTITY`
   * - t
     - :enumerator:`nnp::NeuralNetwork::AF_TANH`
   * - s
     - :enumerator:`nnp::NeuralNetwork::AF_LOGISTIC`
   * - p
     - :enumerator:`nnp::NeuralNetwork::AF_SOFTPLUS`
   * - r
     - :enumerator:`nnp::NeuralNetwork::AF_RELU`
   * - g
     - :enumerator:`nnp::NeuralNetwork::AF_GAUSSIAN`
   * - c
     - :enumerator:`nnp::NeuralNetwork::AF_COS`
   * - S
     - :enumerator:`nnp::NeuralNetwork::AF_REVLOGISTIC`
   * - e
     - :enumerator:`nnp::NeuralNetwork::AF_EXP`
   * - h
     - :enumerator:`nnp::NeuralNetwork::AF_HARMONIC`


----

``normalize_nodes``
^^^^^^^^^^^^^^^^^^^

Activates normalized neural network propagation, i.e. the weighted sum of
connected neuron values is divided by the number of incoming connections
before the activation function is applied. Thus, the default formula to calculate the neuron

.. math::

   y^{k}_{i} = f_a \left( b^{k}_{i} + \sum_{j=1}^{n_l} a^{lk}_{ji} \, y^{l}_{j} \right),

is modified according to:

.. math::

   y^{k}_{i} = f_a \left( \frac{b^{k}_{i} + \sum_{j=1}^{n_l} a^{lk}_{ji} \, y^{l}_{j}}{n_l} \right).

----

``cutoff_type``
^^^^^^^^^^^^^^^^^^^

**Usage**:
   ``cutoff_type <integer> <<float>>``

**Examples**:
   ``cutoff_type 2 0.5``

   ``cutoff_type 7``

Defines the cutoff function type used for all symmetry functions. The first
argument determines the functional form, see :type:`nnp::CutoffFunction::CutoffType`
for all available options. Use one of the following integer numbers to
select the cutoff type. The second argument is optional and sets the parameter
:math:`\alpha`. If not provided, the default value is :math:`\alpha = 0.0`.

.. list-table::
   :header-rows: 1

   * - Cutoff #
     - Cutoff type
   * - 0
     - :enumerator:`nnp::CutoffFunction::CT_HARD`
   * - 1
     - :enumerator:`nnp::CutoffFunction::CT_COS`
   * - 2
     - :enumerator:`nnp::CutoffFunction::CT_TANHU`
   * - 3
     - :enumerator:`nnp::CutoffFunction::CT_TANH`
   * - 4
     - :enumerator:`nnp::CutoffFunction::CT_EXP`
   * - 5
     - :enumerator:`nnp::CutoffFunction::CT_POLY1`
   * - 6
     - :enumerator:`nnp::CutoffFunction::CT_POLY2`
   * - 7
     - :enumerator:`nnp::CutoffFunction::CT_POLY3`
   * - 8
     - :enumerator:`nnp::CutoffFunction::CT_POLY4`


----

``center_symmetry_functions``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``scale_symmetry_functions``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``scale_symmetry_functions_sigma``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combining these keywords determines how the symmetry functions are scaled
before they are used as input for the neural network. See
:type:`nnp::SymmetryFunction::ScalingType` and the following table for allowed
combinations:

.. list-table::
   :header-rows: 1

   * - Keywords present
     - Scaling type
   * - ``None``
     - :enumerator:`nnp::SymmetryFunction::ST_NONE`
   * - ``scale_symmetry_functions``
     - :enumerator:`nnp::SymmetryFunction::ST_SCALE`
   * - ``center_symmetry_functions``
     - :enumerator:`nnp::SymmetryFunction::ST_CENTER`
   * - ``scale_symmetry_functions`` + ``center_symmetry_functions``
     - :enumerator:`nnp::SymmetryFunction::ST_SCALECENTER`
   * - ``scale_symmetry_functions_sigma``
     - :enumerator:`nnp::SymmetryFunction::ST_SCALESIGMA`


----

``scale_min_short``
^^^^^^^^^^^^^^^^^^^

``scale_max_short``
^^^^^^^^^^^^^^^^^^^

**Usage**:
   ``scale_min_short <float>``

   ``scale_max_short <float>``

**Examples**:
   ``scale_min_short 0.0``

   ``scale_max_short 1.0``

Set minimum :math:`S_{\min}` and maximum :math:`S_{\max}` for symmetry function
scaling. See :type:`nnp::SymmetryFunction::ScalingType`.

----

``symfunction_short``
^^^^^^^^^^^^^^^^^^^^^

**Usage**:
   ``symfunction_short <string> <int> ...``

**Examples**:
   ``symfunction_short H 2 H 0.01 0.0 12.0``

   ``symfunction_short H 3 O H 0.2 -1.0 4.0 12.0``

   ``symfunction_short O 9 H H 0.1  1.0 8.0 16.0``

   ``symfunction_short H 12 0.01 0.0 12.0``

   ``symfunction_short O 13 0.1 0.2 1.0 8.0 16.0``

Defines a symmetry function for a specific element. The first argument is the
element symbol, the second sets the the type. The remaining parameters depend on
the symmetry function type, follow the links in the right column of the table
and look for the detailed description of the class.

.. list-table::
   :header-rows: 1

   * - Type integer
     - Symmetry function type
   * - 2
     - :class:`nnp::SymmetryFunctionRadial`
   * - 3
     - :class:`nnp::SymmetryFunctionAngularNarrow`
   * - 9
     - :class:`nnp::SymmetryFunctionAngularWide`
   * - 12
     - :class:`nnp::SymmetryFunctionWeightedRadial`
   * - 13
     - :class:`nnp::SymmetryFunctionWeightedAngular`

