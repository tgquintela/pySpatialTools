
"""
Geo_tools
=========
Package to deal with geodata.


TODO
----
geo_retrieve?
pyproj?
fiona?

"""

from geo_transformations import general_projection, radians2degrees,\
    degrees2radians, ellipsoidal_projection, spheroidal_projection
from geo_filters import check_in_square_area
