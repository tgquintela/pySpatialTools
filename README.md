# pySpatialTools
This package is built in order to provide prototyping tools in python to deal with spatial data in python and model spatial-derived relations between different elements in a system.
In some systems, due to the huge amount of data, the complexity of their topology their local nature or because other practical reasons we are forced to use only local information for model the system properties and dynamics.

pySpatialTools is useful for complex topological systems with different type of spatial data elements and feature data elements in which we are not able to study alls at once because of the data size.

pySpatialTools could be not recommendable for treating some specific problems with homogeneous and/or regular data which could be treated with other python packages, as for example *computational linguistics* ([nltk](http://www.nltk.org/)), *computer vision* or *grid data* ([scipy.ndimage](http://docs.scipy.org/doc/scipy/reference/ndimage.html) and [openCV](https://opencv-python-tutroals.readthedocs.org/en/latest/#)) or others.


# Technical considerations
pySpatialTools is code trying to respect [**PEP8**](https://www.python.org/dev/peps/pep-0008/) python standard style of code and it is structured in the different main modules:
+ Retrieve: module which helps us to code a retrieving methods and use the ones coded in the framework. 
+ Feature_engineering: helps us to transform the spatial relations between elements their local neighborhoods in relations with the element features of all of them.
+ Model: module to code prediction or recommendations tasks.
+ Testing: Testing functions to study how good your model it is regarding different considerations.
+ io: functions which can help to interact with another packages and help to read and output data from pySpatialTools.
+ tests: tests coded in order to ensure the good installation of the package.
+ tools: 
+ utils: 

The main pipeline application in order to match points with features of their neighborhood is:

|   |   |   |   |   |   |
| :------: | :------: | :------: | :------: | :------: | :------: |
| **element** | i | i_s | neighs | neighs_o | neighs_f |
| **topology** | T_0  | T_1 | T_2 | T_3 | T_4 |
| **functions** | ret.input_map | ret | ret.output_map | features._maps_output | - |

The topologies can be repeated or totally different. We can choose the different paths of the pipeline through the choose of the selectors.
It is convenient to design the maps and the selectors.


The main pipeline that the user has to perform is:

1. Build the whole possible retrievers we are going to use.
2. Collect the features and create all the perturbations and aggregations which we could consider needed and the possible output we want.
3. Create the descriptor model we are going to use.
4. Design the selectors in order to be consistent with the retriever-output/descriptor-input.
5. Join altogether in the spatial descriptor model object and compute.


## Main features
* Generic framework to deal easily and quickly with irregular heterogeneous spatial-like data (any data which could be embedded in a topological space).
* Provides a generic interface to code new compatible methods and functions to check their compatibility.
* Support for n-dimensional spaces or any other topological space as we want to define it through the definition of each associated neighborhood for each element using [Retrieve](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve) module.
* Possibility to define irregular neighborhoods regarding on the the elements spatial information, main features of each point or even by combination of simple definitions of retrievers with [collection of retrievers](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/collectionretrievers.py) and a [selector mapper]().
* Some specific simple retrievers coded in [Retrievers](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/retrievers.py) submodule.
* Some specific simple descriptor models coded in [Descriptors](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Feature_engineering/Descriptors) submodule.


## Applications

### Abstract applications
* Transform topological data into another topological space, more appropriate for the study of the system.
* Explore cross-information between aggregated information and punctual features.

### General applications
* Spatial game theory: study and prediction in [*spatial games*]().


## Installation

It could be installed using shell
```shell
git clone https://github.com/tgquintela/pySpatialTools
.\install
```

### Dependencies
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)

## Tutorial

The interface of this package could be summarized with this tools:

* [Discretization](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/Discretization/__init__.py): used to map coordinates of some R^n space into regions.
Example of 2d grid.

```python

grid_size, xlim, ylim = (100, 100), (0, 100), (0, 100)
disc0 = GridSpatialDisc(grid_size, xlim, ylim)
```

* [Region relations](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/Spatial_Relations/__init__.py): used to compute and store relations between regions.
Example of 

```python
disc0 = GridSpatialDisc(grid_size, xlim, ylim)
locs = np.random.random((n, 2))*100

sp_descriptor = (disc0, locs, KRetriever, 5, Countdescriptor)
regionmetrics = CenterLocsRegionDistances()
regionmetrics.compute_distances(sp_descriptor, activated=locs)

```

* [Retriever](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/retrievers.py): used to define neighborhood of spatial elements. Introducing some parameters in the class we are able to define an individualized neighborhood for each spatial element.
Example of retriever k-neighbors.

```python
locs = np.random.random((n, 2))*100
info_ret = np.random.randint(1, 10, n)

ret0 = KRetriever(locs, info_ret, ifdistance=True, relative_pos=diff_vectors)
neighs, relative_positions = ret0.retrieve_neighs(0)

```

* [Descriptor](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Feature_engineering/descriptormodel.py#L11): used to measure a descriptors of each desired and defined element in order to model the system.
It consider the generic case in which is defined by the own point features of the element considered and the point features and the relative
spatial situation of each neighborhood element.

```python
gret = CollectionRetrievers(retrievers_list)
map_vals_i = create_mapper_vals_i(type_elements_corr)
feats_ret = FeaturesRetriever(features_list)
desc = Countdescriptor(feats_ret, map_vals_i)
sp_model = SpatialDescriptorModel(gret, desc)
sp_corr = sp_model.compute_nets()

```

* Aggregation: used to compute descriptors for regions by aggregating the point features of the defined region neighborhood.

```python
grid_size, xlim, ylim = (100, 100), (0, 100), (0, 100)
disc0 = GridSpatialDisc(grid_size, xlim, ylim)
locs = np.random.random((n, 2))*100
sp_descriptor = (disc0, locs, KRetriever, 5, Countdescriptor)
regionmetrics = CenterLocsRegionDistances()
regionmetrics.compute_distances(sp_descriptor, activated=locs)
aggcharacs, u_regs, null_vals = countdesc.compute_aggdescriptors(disc, regionmetrics, locs)
```

## Testing

You need to ensure that all the package requirements are installed. pySpatialTools provide a testing module which could be called by importing the module and applying test function.
If we are in the python idle:

```python
import pySpatialTools
pySpatialTools.test()
```
or from the shell
```shell
>> python -c "import pySpatialTools; pySpatialTools.test()"

***---*** Testing python package pySpatialTools ***---***
---------------------------------------------------------
Test compared with a reference computer with specs:
Linux Mint 17 Qiana
Kernel Linux 3.13.0-24-generic
Intel Core i7-3537U
4GB de RAM
NVidia GeForce GT720M 2GB
-------------------------------
Average time in ref. computer: 20.58 seconds.
Time testing in this computer: 20.83 seconds.

```

for developers you can test from the source using nosetests of nose package.

```shell
nosetests path/to/dir
```

## Project and contributions
This package is in an early stage. If there is any idea to improve it or even code do not hesitate to do it or communicate with the main developer through mail:
tgq.spm@gmail.com


### Next steps
- [x] Format package into some conventions.
- [ ] Code Testing module.
- [ ] Prediction module.
- [ ] Examples

### License
pySpatialTools is available as open source under the terms of the [MIT License](https://github.com/tgquintela/pySpatialTools/blob/master/LICENSE).

