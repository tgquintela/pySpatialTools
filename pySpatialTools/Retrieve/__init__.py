
"""
Retrive Module
==============
Module oriented to group functions related with the retrieve of the
neighbourhood or local properties related with the neighbourhood.


Space discretization
--------------------
http://www.esri.com/library/whitepapers/pdfs/shapefile.pdf
https://en.wikipedia.org/wiki/Shapefile
arcpy



"""

# import neighbourhood
from neighbourhood import Neighbourhood

# import retrievers
from retrievers import KRetriever, CircRetriever, SameRegionRetriever

# import spatial discretizors
from circdiscretization import CircularSpatialDisc
from bisectordiscretization import BisectorSpatialDisc
from griddiscretization import GridSpatialDisc
from polygondiscretization import IrregularSpatialDisc
