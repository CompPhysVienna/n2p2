.. _descriptors:

Atomic environment descriptors
==============================

Multiple atomic environment descriptors (symmetry functions) are already
implemented. Use them via the ``symfunction_short`` keyword (see
:ref:`this<keywords>` page).

Original atom-centered symmetry functions
-----------------------------------------

Taken from the original 2007 paper

`J. Behler and M. Parrinello, Phys. Rev. Lett. 98, 146401 (2007)
<https://doi.org/10.1103/PhysRevLett.98.146401>`_

these are the basic radial and angular symmetry functions:


* Radial symmetry function (:cpp:class:`nnp::SymFncExpRad`):

  .. math::

     G^2_i = \sum_{j \neq i} \mathrm{e}^{-\eta(r_{ij} - r_s)^2} f_c(r_{ij})

* Angular symmetry function (:cpp:class:`nnp::SymFncExpAngn`):

  .. math::

     G^3_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
             \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
             \mathrm{e}^{-\eta( r_{ij}^2 + r_{ik}^2 + r_{jk}^2 ) }
             f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk})

Atom-centered symmetry functions (continued)
--------------------------------------------

In 2011 more symmetry functions were presented here:

`J. Behler, J. Chem. Phys. 134, 074106 (2011) <http://dx.doi.org/10.1063/1.3553717>`_

Amongst others a variant of the above angular symmetry function was introduced:


* Modified angular symmetry function (:cpp:class:`nnp::SymFncExpAngw`, for
  historic reasons the type number here is 9):

  .. math::

     G^9_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
             \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
             \mathrm{e}^{-\eta( r_{ij}^2 + r_{ik}^2 ) } f_c(r_{ij}) f_c(r_{ik})

.. note::

   The implementation of both angular symmetry functions described above also
   supports an additional shift parameter :math:`r_s` in the exponential function
   (analogous to the radial symmetry function or the weighted symmetry functions
   below). Specifying the shift parameter is optional, omitting it will reproduce
   the formulae given above.

Weighted atom-centered symmetry functions
-----------------------------------------

In 2018 an approach to overcome limitiations for systems with many different
atom species was put forward here:

`M. Gastegger, L. Schwiedrzik, M. Bittermann, F. Berzsenyi and P. Marquetand,
J. Chem. Phys. 148, 241709 (2018) <https://doi.org/10.1063/1.5019667>`_

Here two variants of the original symmetry functions were presented:


* Weighted radial symmetry function (:cpp:class:`nnp::SymFncExpRadWeighted`)

  .. math::

     G^{12}_i = \sum_{j \neq i} Z_j \,
                \mathrm{e}^{-\eta(r_{ij} - r_s)^2}
                f_c(r_{ij})

* Weighted angular symmetry function (:cpp:class:`nnp::SymFncExpAngnWeighted`)

  .. math::

     G^{13}_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
                Z_j Z_k \,
                \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
                \mathrm{e}^{-\eta \left[
                (r_{ij} - r_s)^2 + (r_{ik} - r_s)^2 + (r_{jk} - r_s)^2 \right] }
                f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk})

.. _polynomial_sf:

Low-cost polynomial symmetry functions with compact support
-----------------------------------------------------------

In 2020 a new set of computationally efficient symmetry functions was proposed
here:

`Bircher, M. P.; Singraber, A.; Dellago, C., arXiv:2010.14414 [cond-mat,
physics:physics] (2020). <http://arxiv.org/abs/2010.14414>`__

In contrast to the above definitions these **polynomial** symmetry functions do
not require the computation of expensive exponential terms because they are
based solely on polynomial window functions in the radial and angular domain. As
shown in the publication a significant increase in performance ensues without
sacrificing descriptive power. Furthermore, they are well suited to describe
complex atomic environments because their angular sensitivity can be easily
controlled via the free hyperparameters. The following variants of polynomial
symmetry functions are implemented (here :math:`C(x, x_\text{low},
x_\text{high})` is a function with compact support :math:`\left[x_\text{low},
x_\text{high}\right]`):

* Radial symmetry function (:cpp:class:`nnp::SymFncCompRad`)

  .. math::

     G^{20}_i = \sum_{\substack{j \neq i}} C(r_{ij}, r_l, r_c)

* Angular symmetry function, narrow variant (:cpp:class:`nnp::SymFncCompAngn`)

  .. math::

     G^{21}_i = \sum_{\substack{j,k\neq i \\ j < k}} C(r_{ij}, r_l, r_c)
                C(r_{ik}, r_l, r_c) C(r_{jk}, r_l, r_c)
                C(\theta_{ijk}, \theta_l, \theta_r)

* Angular symmetry function, wide variant (:cpp:class:`nnp::SymFncCompAngw`)

  .. math::

     G^{22}_i = \sum_{\substack{j,k\neq i \\ j < k}} C(r_{ij}, r_l, r_c)
                C(r_{ik}, r_l, r_c) C(\theta_{ijk}, \theta_l, \theta_r)

* Weighted radial symmetry function (:cpp:class:`nnp::SymFncCompRadWeighted`)

  .. math::

     G^{23}_i = \sum_{\substack{j \neq i}} Z_j C(r_{ij}, r_l, r_c)

* Weighted angular symmetry function, narrow variant
  (:cpp:class:`nnp::SymFncCompAngnWeighted`)

  .. math::

     G^{24}_i = \sum_{\substack{j,k\neq i \\ j < k}} Z_j Z_k
                C(r_{ij}, r_l, r_c) C(r_{ik}, r_l, r_c)
                C(r_{jk}, r_l, r_c) C(\theta_{ijk}, \theta_l, \theta_r),

* Weighted angular symmetry function, wide variant
  (:cpp:class:`nnp::SymFncCompAngwWeighted`)

  .. math::

     G^{25}_i = \sum_{\substack{j,k\neq i \\ j < k}} Z_j Z_k
                C(r_{ij}, r_l, r_c) C(r_{ik}, r_l, r_c)
                C(\theta_{ijk}, \theta_l, \theta_r),
