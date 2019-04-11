#!/usr/bin/env python

import numpy as np
import sys
import itertools

symfunc_list_radial = [{"type": 2, "element": 'Cu', "neighbors": {'Cu'}},
                       {"type": 2, "element": 'Cu', "neighbors": {'S'}},
                       {"type": 2, "element": 'S', "neighbors": {'Cu'}},
                       {"type": 2, "element": 'S', "neighbors": {'S'}}]


symfunc_list_angular_narrow = [{"type": 3, "element": 'Cu', "neighbors": {'Cu'}},
                               {"type": 3, "element": 'Cu', "neighbors": {'S'}},
                               {"type": 3, "element": 'Cu', "neighbors": {'Cu', 'S'}},
                               {"type": 3, "element": 'S', "neighbors": {'Cu'}},
                               {"type": 3, "element": 'S', "neighbors": {'S'}},
                               {"type": 3, "element": 'S', "neighbors": {'Cu', 'S'}}]



symfunc_list = symfunc_list_radial + symfunc_list_angular_narrow

# get all unique symmetry function types contained in the list of symmetry functions
sf_types = set([sf['type'] for sf in symfunc_list])
print(sf_types)

# get all unique elements contained in the list of symmetry functions
elems = set([el['element'] for el in symfunc_list])
print(elems)

# get all unique unordered pairs of two elements (including pairs of two elements that are the same)
elem_pairs = itertools.combinations_with_replacement(elems, 2)


print()

sf_list_type_2 = [sf for sf in symfunc_list if sf['type']==2]
for central_elem in elems:
    sys.stdout.write("# Radial symmetry functions for element {0:2s}\n".format(central_elem))
    sf_list_central = [sf for sf in sf_list_type_2 if sf['element']==central_elem]
    for neighbor_elem in elems:
        sf_list_neighbors = [sf for sf in sf_list_central if sf['neighbor']==neighbor_elem]
        print(sf_list_neighbors)
        sys.stdout.write('\n')


# sf_list_type_3 = [sf for sf in symfunc_list if sf['type']==3]
# for central_elem in elems:
#     sys.stdout.write("# Narrow angular symmetry functions for element {0:2s}\n".format(central_elem))
#     sf_list_central = [sf for sf in sf_list_type_3 if sf['element']==central_elem]
#     for neighbor_elem_1 in elems:
#         sf_list_neighbors_1 = [sf for sf in sf_list_central if sf['neighbor_1']==neighbor_elem_1]
#         for neighbor_elem_2 in elems:
#             sf_list_neighbors_2 = [sf for sf in sf_list_central if sf['neighbor_2'] == neighbor_elem_2]
#             print(sf_list_neighbors_2)
#             sys.stdout.write('\n')