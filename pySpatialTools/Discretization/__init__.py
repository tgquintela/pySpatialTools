
"""
Discretization
==============
Space discretization module groups functions to discretize space in regions
and facilitate the retrieve by regions or define neighbourhood with fixed
regions. These modules groups precoded functions and interactors with other
packages in order to discretize a space in a given way.
They can be used to do Spatial Hashing.

"""
### Import spatial discretizors
## 2D Discretization
from Discretization_2d.circdiscretization import CircularInclusiveSpatialDisc,\
    CircularExcludingSpatialDisc
from Discretization_2d.bisectordiscretization import BisectorSpatialDisc
from Discretization_2d.griddiscretization import GridSpatialDisc
from Discretization_2d.polygondiscretization import IrregularSpatialDisc

## Set Discretization
from Discretization_set import SetDiscretization
from spatialdiscretizer import BaseSpatialDiscretizor

## Discretization information parsers
from aux_discretization_parsing import _discretization_parsing_creation,\
    _discretization_regionlocs_parsing_creation,\
    _discretization_information_creation
