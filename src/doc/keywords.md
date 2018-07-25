NNP configuration: keywords
===========================

The NNP settings file (usually named `input.nn`) contains the setup of neural
networks and symmetry functions. Each line may contain a single keyword with no,
one or multiple arguments. Keywords and arguments are separated by at least one
whitespace. Lines or part of lines can be commented out with the symbol "#",
everything right of "#" will be ignored. The order of keywords is not
important, most keywords may only appear once (exceptions: `symfunction_short`
and `atom_energy`). Here is the list of available keywords (if no usage
information is provided, then the keyword does not support any arguments).

General NNP keywords
--------------------

These keywords provide the basic setup for a neural network potential.
Therefore, they are mandatory for most NNP applications and a minimal example
setup file would contain at least:

```
number_of_elements 1
elements H
global_hidden_layers_short 2
global_nodes_short 10 10
global_activation_short t t l
cutoff_type 1
symfunction_short ...
...
```
These commands will set up a neural network potential for hydrogen atoms (2
hidden layers with 10 neurons each, hyperbolic tangent activation function) and
the cosine cutoff function for all symmetry functions. Of course, further
symmetry functions need to be specified by adding multiple lines starting with
the `symfunction_short` keyword.

---------------------------

### `number_of_elements`

*Usage:* <code><b>number_of_elements</b> <i>integer</i></code>

*Example:* <code><b>number_of_elements</b> <i>3</i></code>

Defines the number of elements the neural network potential is designed for.

---------------------------

### `elements`

*Usage:* <code><b>elements</b> <i>string <...></i></code>

*Example:* <code><b>elements</b> <i>O H Zn</i></code>

This keyword defines all elements via a list of element symbols. The number
of items provided has to be consistent with the argument of the
`number_of_elements` keyword. The order of the items is not important,
elements are automatically sorted according to their atomic number. The list
nnp::ElementMap::knownElements contains a list of recognized element symbols.

---------------------------

### `atom_energy`

*Usage:* <code><b>atom_energy</b> <i>string float</i></code>

*Example:* <code><b>atom_energy</b> <i>O -74.94518524</i></code>

Definition of atomic reference energy. Shifts the total potential energy by
the given value for each atom of the specified type. The first argument is the
element symbol, the second parameter is the shift energy.

---------------------------

### `global_hidden_layers_short`

*Usage:* <code><b>global_hidden_layers_short</b> <i>integer</i></code>

*Example:* <code><b>global_hidden_layers_short</b> <i>3</i></code>

Sets the number of hidden layers for neural networks of all elements.

---------------------------

### `global_nodes_short`

*Usage:* <code><b>global_nodes_short</b> <i>integer <...></i></code>

*Example:* <code><b>global_nodes_short</b> <i>15 10 5</i></code>

Sets the number of neurons in the hidden layers of neural networks of all
elements. The number of integer arguments has to be consistent with the
argument of the `global_hidden_layers_short` keyword. Note: The number of
input layer neurons is determined by the number of symmetry functions and there
is always only a single output neuron.

---------------------------

### `global_activation_short`

*Usage:* <code><b>global_activation_short</b> <i>char char <...></i></code>

*Example:* <code><b>global_activation_short</b> <i>t t t l</i></code>

Sets the activation function per layer for hidden layers and output layers
of neural networks of all elements. The number of integer arguments has to
be consistent with the argument of the `global_hidden_layers_short` keyword
(i.e. number of hidden layers + 1). Activation functions are chosen via
single characters from the following table (see also
nnp::NeuralNetwork::ActivationFunction).

| Character | Activation function type        |
|:---------:| ------------------------------- |
|         l | nnp::NeuralNetwork::AF_IDENTITY |
|         t | nnp::NeuralNetwork::AF_TANH     |
|         s | nnp::NeuralNetwork::AF_LOGISTIC |
|         p | nnp::NeuralNetwork::AF_SOFTPLUS |

---------------------------

### `normalize_nodes`

Activates normalized neural network propagation, i.e. the weighted sum of
connected neuron values is divided by the number of incoming connections
before the activation function is applied. Thus, the default formula to calculate the neuron @f$y^{k}_{i}@f$ in layer @f$k@f$,
@f[
y^{k}_{i} = f_a \left( b^{k}_{i} + \sum_{j=1}^{n_l} a^{lk}_{ji} \, y^{l}_{j} \right),
@f]
is modified according to:
@f[
y^{k}_{i} = f_a \left( \frac{b^{k}_{i} + \sum_{j=1}^{n_l} a^{lk}_{ji} \, y^{l}_{j}}{n_l} \right).
@f]

---------------------------

### `cutoff_type`

*Usage:* <code><b>cutoff_type</b> <i>integer &lt;float&gt;></i></code>

*Example:* <code><b>cutoff_type</b> <i>2 0.5</i></code>

Defines the cutoff function type used for all symmetry functions. The first
argument determines the functional form, see nnp::CutoffFunction::CutoffType
for all available options. Use one of the following integer numbers to
select the cutoff type. The second argument is optional and sets the parameter
@f$\alpha@f$. If not provided, the default value is @f$\alpha = 0.0@f$.

| Cutoff # | Cutoff type                   |
|:--------:| ----------------------------- |
|        0 | nnp::CutoffFunction::CT_HARD  |
|        1 | nnp::CutoffFunction::CT_COS   |
|        2 | nnp::CutoffFunction::CT_TANHU |
|        3 | nnp::CutoffFunction::CT_TANH  |
|        4 | nnp::CutoffFunction::CT_EXP   |
|        5 | nnp::CutoffFunction::CT_POLY1 |
|        6 | nnp::CutoffFunction::CT_POLY2 |
|        7 | nnp::CutoffFunction::CT_POLY3 |
|        8 | nnp::CutoffFunction::CT_POLY4 |

---------------------------

### `center_symmetry_functions`
### `scale_symmetry_functions`
### `scale_symmetry_functions_sigma`

Combining these keywords determines how the symmetry functions are scaled
before they are used as input for the neural network. See
nnp::SymmetryFunction::ScalingType and the following table for allowed
combinations:

| Keywords present                                         | Scaling type                          |
| -------------------------------------------------------- | ------------------------------------- |
| `None`                                                   | nnp::SymmetryFunction::ST_NONE        |
| `scale_symmetry_functions`                               | nnp::SymmetryFunction::ST_SCALE       |
| `center_symmetry_functions`                              | nnp::SymmetryFunction::ST_CENTER      |
| `scale_symmetry_functions` + `center_symmetry_functions` | nnp::SymmetryFunction::ST_SCALECENTER |
| `scale_symmetry_functions_sigma`                         | nnp::SymmetryFunction::ST_SCALESIGMA  |

---------------------------

### `scale_min_short`
### `scale_max_short`

*Usage:* <code><b>scale_min_short</b> <i>float</i></code>

*Usage:* <code><b>scale_max_short</b> <i>float</i></code>

*Example:* <code><b>scale_min_short</b> <i>0.0</i></code>

*Example:* <code><b>scale_max_short</b> <i>1.0</i></code>

Set minimum @f$S_{\min}@f$ and maximum @f$S_{\max}@f$ for symmetry function
scaling. See nnp::SymmetryFunction::ScalingType.

---------------------------

### `symfunction_short`

*Usage:* <code><b>symfunction_short</b> <i>string int ...</i></code>

*Example:* <code><b>symfunction_short</b> <i>H 2 H 0.01 0.0 12.0</i></code>

*Example:* <code><b>symfunction_short</b> <i>H 3 O H 0.2 -1.0 4.0 12.0</i></code>

*Example:* <code><b>symfunction_short</b> <i>O 9 H H 0.1  1.0 8.0 16.0</i></code>

*Example:* <code><b>symfunction_short</b> <i>H 12 0.01 0.0 12.0</i></code>

*Example:* <code><b>symfunction_short</b> <i>O 13 0.1 0.2 1.0 8.0 16.0</i></code>

Defines a symmetry function for a specific element. The first argument is the
element symbol, the second sets the the type. The remaining parameters depend on
the symmetry function type, follow the links in the right column of the table
and look for the detailed description of the class.

| Type integer | Symmetry function type               |
|:------------:| ------------------------------------ |
|            2 | nnp::SymmetryFunctionRadial          |
|            3 | nnp::SymmetryFunctionAngularNarrow   |
|            9 | nnp::SymmetryFunctionAngularWide     |
|           12 | nnp::SymmetryFunctionWeightedRadial  |
|           13 | nnp::SymmetryFunctionWeightedAngular |

---------------------------

Additional training keywords
----------------------------

The NNP tool `nnp-train` allows for the training of a new neural network
potential. A variety of training options can be specified by the additional
keywords listed below. These keywords will be ignored by all other NNP tools.

---------------------------

### `epochs`

*Usage:* <code><b>epochs</b> <i>integer</i></code>

*Example:* <code><b>epochs</b> <i>100</i></code>

Sets how many epochs of neural network training are performed. Usually, to
train a neural network for one epoch means that every piece of information in
the training data set is used once to update the weights. Here, the actual
number of updates and the procedure how update information is
chosen depends on the keywords `short_energy_fraction`,
`short_force_fraction`, `repeated_energy_update,`
`selection_mode`, `short_energy_error_threshold`,
`short_force_error_threshold` and `rmse_threshold_trials`.

---------------------------

### `updater_type`

*Usage:* <code><b>updater_type</b> <i>integer</i></code>

*Example:* <code><b>updater_type</b> <i>1</i></code>

Determines which algorithm is used for updating weights. Currently these
options are available, use the type integer as keyword argument (see
nnp::Training::UpdaterType).

| Type integer | Weight update algorithm           |
|:------------:| --------------------------------- |
|            0 | nnp::Training::UT_GRADIENTDESCENT |
|            1 | nnp::Training::UT_KALMANFILTER    |

Note: the gradient descent method is implemented only for reference. Successful
training of a neural network potential is not to be expected.

---------------------------

### `parallel_mode`

*Usage:* <code><b>parallel_mode</b> <i>integer</i></code>

*Example:* <code><b>parallel_mode</b> <i>1</i></code>

nnp::Training::ParallelMode

| Type integer | Parallelization mode          |
|:------------:| ----------------------------- |
|            0 | nnp::Training::PM_SERIAL      |
|            1 | nnp::Training::PM_MULTISTREAM |

---------------------------

### `update_strategy`

*Usage:* <code><b>update_strategy</b> <i>integer</i></code>

*Example:* <code><b>update_strategy</b> <i>1</i></code>

nnp::Training::UpdateStrategy

| Type integer | Weight update strategy     |
|:------------:| -------------------------- |
|            0 | nnp::Training::US_COMBINED |
|            1 | nnp::Training::US_ELEMENT  |

---------------------------

### `selection_mode`

*Usage:* <code><b>selection_mode</b> <i>integer</i></code>

*Example:* <code><b>selection_mode</b> <i>1</i></code>

nnp::Training::SelectionMode

| Type integer | Update candidate selection mode |
|:------------:| ------------------------------- |
|            0 | nnp::Training::SM_RANDOM        |
|            1 | nnp::Training::SM_SORT          |
|            2 | nnp::Training::SM_THRESHOLD     |
