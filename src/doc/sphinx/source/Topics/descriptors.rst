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


* 
  Radial symmetry function (:class:`nnp::SymmetryFunctionRadial`):

  .. math::

     G^2_i = \sum_{j \neq i} \mathrm{e}^{-\eta(r_{ij} - r_s)^2} f_c(r_{ij}) 

* 
  Angular symmetry function (:class:`nnp::SymmetryFunctionAngularNarrow`):

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


* Modified angular symmetry function (:class:`nnp::SymmetryFunctionAngularWide`, for
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


* 
  Weighted radial symmetry function (:class:`nnp::SymmetryFunctionWeightedRadial`)

  .. math::

     G^{12}_i = \sum_{j \neq i} Z_j \,
                \mathrm{e}^{-\eta(r_{ij} - r_s)^2}
                f_c(r_{ij}) 

* 
  Weighted angular symmetry function (:class:`nnp::SymmetryFunctionWeightedAngular`)

  .. math::

     G^{13}_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
                Z_j Z_k \,
                \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
                \mathrm{e}^{-\eta \left[
                (r_{ij} - r_s)^2 + (r_{ik} - r_s)^2 + (r_{jk} - r_s)^2 \right] }
                f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk}) 
