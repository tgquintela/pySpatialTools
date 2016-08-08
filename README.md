[![Build Status](https://travis-ci.org/tgquintela/pySpatialTools.svg?branch=master)](https://travis-ci.org/tgquintela/pySpatialTools)
[![Coverage Status](https://coveralls.io/repos/github/tgquintela/pySpatialTools/badge.svg?branch=master)](https://coveralls.io/github/tgquintela/pySpatialTools?branch=master)
# pySpatialTools
This package is built in order to provide prototyping tools in python to deal with spatial data in python and model spatial-derived relations between different elements in a system.
In some systems, due to the huge amount of data, the complexity of their topology their local nature or because other practical reasons we are forced to use only local information for model the system properties and dynamics, or for getting insightful local-derived features.

pySpatialTools is useful for complex topological systems with different type of spatial data elements and feature data elements in which we are not able to study all at once because of the data size.

pySpatialTools could be not recommendable for treating some specific problems with homogeneous and/or regular data which could be treated with other python packages, as for example *computational linguistics* ([nltk](http://www.nltk.org/)), *computer vision* or *grid data* ([scipy.ndimage](http://docs.scipy.org/doc/scipy/reference/ndimage.html) and [openCV](https://opencv-python-tutroals.readthedocs.org/en/latest/#)) or others.


## Technical considerations
pySpatialTools is a framework which is coded trying to respect [**PEP8**](https://www.python.org/dev/peps/pep-0008/) python standard style of code.
As a framework, pySpatialTools is not specifically case-oriented but flexible which allows the user to easy integrate their tools, specific data and problem to the framework in order to ease the solution of their problem. The framework not only provides high-level tools and general useful functions for spatial problems, but also a structures and standards that structure the program in an understandable way.

The main structure of the problem is inspired by a general definition of the problem of *spatial analysis*. The data is divided in spatial data and features data. The work-flow is:

1. Retrieve of elements using their spatial information.
2. Get the associated features of the retrieved spatial elements.
3. Compute the descriptors of the elements by using the neighborhood information.
4. Aggregate to a final computed magnitude.

There are some *hot spots*, where the user can take profit of the framework code and use it as a wrapper of his own code in order to take profit of the tools he has available. The main *hot spots* are:
* `BaseRetriever` (or even its son classes): where it happens the connection (and possibly the storage) of the spatial data, as well as the retrieve.
* `BaseFeatures` (or even its son classes): where it happens the connection (and possibly the storage) of the feature data, as well as the retrieve and computation of descriptors).
* `BaseDescriptorModel`: where it is computed the descriptors of the selected element from the associated spatial neighborhood retrieved.
* `BasePerturbations`: where it is implemented the statistical random models to apply to the data in order to perform posterior statistical testing.
* `BaseRelativePositioner`: where it is implemented the basic relative position quantity (distance, similarity or other complex object) definition for assign a posteriori measure between elements and the elements in their neighborhoods after the retrieve. 

With these classes the users, by using inheritance and respecting the standards of the framework, can adapt their tools to the framework in order to save time and efforts, and give a common interface.


## Main features
* Generic framework to deal easily and quickly with *irregular heterogeneous spatial-like data* (any data which could be embedded in a topological space).
* Provides a generic interface to code new compatible methods and functions to check their compatibility.
* Generic support for n-dimensional metric spaces or other topological spaces by defining to each element their neighbourhood using [Retrieve](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve) module.
* Spatial data could be points, lines, polygons or other complex elements in n-dimensional spaces.
* Features data could be sparse defined with key, values possible definition.
* Possibility to define irregular neighborhoods regarding on the the elements spatial information, main features of each point or even by combination of simple definitions of retrievers with [collection of retrievers](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/collectionretrievers.py) and a [selector mapper](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/utils/selectors/spdesc_mapper.py).
* Possibility to wrap an interaction with external spatial databases.
* Some specific simple retrievers (as K-order, k-neighbour, radius retriever) coded in [Retrievers](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/retrievers.py) submodule.
* Some specific simple descriptor models (as histogram or averaging desc) coded in [Descriptors](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Feature_engineering/Descriptors) submodule.


## Applications

### Abstract applications
* Transform topological data into another topological space, more appropriate for the study of the system.
* Explore cross-information between aggregated information and punctual features.
* Assigning local features to some elements for posterior study.

### General applications
* Spatial game theory: study and prediction in [*spatial games*]() evolution by using the statistical and machine learning tools to the extracted features.
* Spatial regression models: perform a local regression model for a heterogeneous and user-defined neighborhood.
* Support tool in Exploratory Analysis computing quickly easy local measures.
* Preprocessing information for modeling or visualization, e.g. spatially aggregating data.
* Study the spatial scaling property of features data.

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
Average time in ref. computer: 3.50 seconds.
Time testing in this computer: 3.83 seconds.

```

for developers you can test from the source using nosetests of nose package.

```shell
nosetests path/to/dir
```

## Tutorial
For know how to use the program, check the [tutorial](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/TUTORIAL.md).

## Project and contributions
This package is in an early stage. If there is any idea to improve it or even code do not hesitate to do it or communicate with the main developer through mail:
tgq.spm@gmail.com


### Next steps
- [x] Format package into some conventions.
- [x] Examples
- [ ] Code Tester module.
- [ ] Prediction module.

Other [TODOs](https://github.com/tgquintela/pySpatialTools/blob/master/TODO.md)

### License
pySpatialTools is available as open source under the terms of the [MIT License](https://github.com/tgquintela/pySpatialTools/blob/master/LICENSE).

