# pySpatialTools
Tools to deal with spatial data in python. This package provides prototyping tools for spatial data modelling.
In spatial data, the information of the environment is important to understand, predict and take profit of the information.
pySpatialTools is a framework mainly built in order to obtain spatial descriptors of elements or points embedded in a n-dimensional space.



# Main features
* Generic framework to deal easily and quickly with spatial data.
* Support of n-dimensional spaces.
* Some specific methods coded.
* Provides a generic interface to code compatible methods.
* Possibility to define irregular neighborhoods regarding on the position of the points coordinates or main features of each point.
* Explore cross-information between aggregated information and punctual features.


# Installation

It could be installed using shell
```shell
git clone https://github.com/tgquintela/pySpatialTools
.\install
```

# Dependencies
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)

# Tutorial

The interface of this package could be summarized with this tools:

* [Discretization](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/Discretization/__init__.py): used to map coordinates of some R^n space into regions.
Example of 2d grid.

```python
from pySpatialTools.Retrieve.Discretization import GridSpatialDisc

grid_size, xlim, ylim = (100, 100), (0, 100), (0, 100)
disc0 = GridSpatialDisc(grid_size, xlim, ylim)
```

* [Region relations](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/Spatial_Relations/__init__.py): used to compute and store relations between regions.
Example of 

```python
from pySpatialTools.Feature_engineering.count_descriptor import Countdescriptor
from pySpatialTools.Retrieve import KRetriever
from pySpatialTools.Retrieve.Spatial_Relations.regionmetrics import CenterLocsRegionDistances

disc0 = GridSpatialDisc(grid_size, xlim, ylim)
locs = np.random.random((n, 2))*100

sp_descriptor = (disc0, locs, KRetriever, 5, Countdescriptor)
regionmetrics = CenterLocsRegionDistances()
regionmetrics.compute_distances(sp_descriptor, activated=locs)

```

* [Retriever](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/retrievers.py): used to define neighborhood of spatial elements. Introducing some parameters in the class we are able to define an individualized neighborhood for each spatial element.
Example of retriever k-neighbors.

```python
from pySpatialTools.Retrieve.relative_positioner import diff_vectors
from pySpatialTools.Retrieve import KRetriever

locs = np.random.random((n, 2))*100
info_ret = np.random.randint(1, 10, n)

ret0 = KRetriever(locs, info_ret, ifdistance=True, relative_pos=diff_vectors)
neighs, relative_positions = ret0.retrieve_neighs(0)

```

* [Descriptor](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Feature_engineering/descriptormodel.py#L11): used to measure a descriptors of each desired and defined element in order to modelling the system.
It consider the generic case in which is defined by the own point features of the element considered and the point features and the relative
spatial situation of each neighborhood element.

```python

```

* Aggregation: used to compute descriptors for regions by aggregating the point features of the defined region neighborhood.

```python
from pySpatialTools.Feature_engineering.count_descriptor import Countdescriptor
from pySpatialTools.Retrieve import KRetriever
from pySpatialTools.Retrieve.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve.Spatial_Relations.regionmetrics import CenterLocsRegionDistances

grid_size, xlim, ylim = (100, 100), (0, 100), (0, 100)
disc0 = GridSpatialDisc(grid_size, xlim, ylim)
locs = np.random.random((n, 2))*100
sp_descriptor = (disc0, locs, KRetriever, 5, Countdescriptor)

regionmetrics = CenterLocsRegionDistances()
regionmetrics.compute_distances(sp_descriptor, activated=locs)

aggcharacs, u_regs, null_vals = countdesc.compute_aggdescriptors(disc, regionmetrics, locs)
```

# Testing





