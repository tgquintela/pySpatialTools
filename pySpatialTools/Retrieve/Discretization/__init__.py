
"""
Space discretization
====================
Space discretization module groups functions to discretize space in regions
and facilitate the retrieve by regions or define neighbourhood with fixed
regions.


TODO
----
http://www.esri.com/library/whitepapers/pdfs/shapefile.pdf
https://en.wikipedia.org/wiki/Shapefile
arcpy

"""

# import spatial discretizors
from circdiscretization import CircularSpatialDisc
from bisectordiscretization import BisectorSpatialDisc
from griddiscretization import GridSpatialDisc
from polygondiscretization import IrregularSpatialDisc

# import special functions
from polygondiscretization import fit_polygondiscretizer

