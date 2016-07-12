
# Tutorial

## Structure of the code

pySpatialTools is structured in the different main modules:
+ Retrieve: module which helps us to code a retrieving methods and use the ones coded in the framework to perform spatial retrieving. 
+ FeatureManagement: helps us to transform the spatial relations between elements their local neighborhoods in relations with the element features of all of them.
+ SpatialRelations: module to help us to compute and store spatial relations.
+ Discretization: tools to discretize the space, to assign element points to element regions or to assign element to high group entity elements.
+ Preprocess: tools to transform spatial and feature data in order to prepare for posterior tasks.
+ io: functions which can help to interact with another packages and help to read and output data from pySpatialTools.
+ tests: tests coded in order to ensure the good installation and performance of the package.
+ tools: util functions to be used by the user.
+ utils: util functions used by other modules of the package.


## Theoretical usage
The main pipeline application in order to match points with features of their neighborhood is:

|   |   |   |   |   |   |
| :------: | :------: | :------: | :------: | :------: | :------: |
| **element** | i | i_s | neighs | neighs_o | neighs_f |
| **topology** | T_0  | T_1 | T_2 | T_3 | T_4 |
| **functions** | ret.input_map | ret | ret.output_map | features._maps_output | - |

The topologies can be repeated or totally different. We can choose the different paths of the pipeline through the choose of the selectors.
It is convenient to design the maps and the selectors.


## Usage

The main pipeline that the user has to perform is:

> 1. Build the whole possible retrievers we are going to use.
> 2. Collect the features and create all the perturbations and aggregations which we could consider needed and the possible output we want.
> 3. Ensure the indexes of the spatial elements are properly assigned to the indexes of the features data.
> 4. Create the descriptor model we are going to use.
> 5. Design the selectors in order to be consistent with the retriever-output/descriptor-input.
> 6. Join altogether in the spatial descriptor model object and compute.


## Code usage
The interface of this package could be summarized with this tools:

* [Discretization](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/Discretization/__init__.py): used to map coordinates of some R^n space into regions.

Example of 2d grid.
```python
from pySpatialTools.Discretization import GridSpatialDisc
import numpy as np

grid_size, xlim, ylim = (100, 100), (0, 100), (0, 100)
disc0 = GridSpatialDisc(grid_size, xlim, ylim)
locations = np.random.random((10000, 2))*100

gridregions = disc0.discretize(locations)

```

* [Region relations](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/Spatial_Relations/__init__.py): used to compute and store relations between regions.

```python
from pySpatialTools.SpatialRelations import RegionDistances

mainmapper = RegionDistances(relations=relations, _data=_data, symmetric=symmetric)

```

* [Retriever](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Retrieve/retrievers.py): used to define neighborhood of spatial elements. Introducing some parameters in the class we are able to define an individualized neighborhood for each spatial element.
Example of retriever k-neighbors.

```python
import numpy as np
from pySpatialTools.Retrieve import KRetriever

locs = np.random.random((n, 2))*100
info_ret = np.random.randint(1, 10, n)

ret0 = KRetriever(locs, info_ret, ifdistance=True, relative_pos=diff_vectors)
neighs, relative_positions = ret0.retrieve_neighs(0)
```

* [Descriptor](https://github.com/tgquintela/pySpatialTools/blob/master/pySpatialTools/Feature_engineering/descriptormodel.py#L11): used to measure a descriptors of each desired and defined element in order to model the system.
It consider the generic case in which is defined by the own point features of the element considered and the point features and the relative spatial situation of each neighborhood element. It is integrated in the feature data interaction part following the model of in-data computation.
This fact make it highly interactive with the `Features` classes and the `FeatureManager` class.
The main tasks that the `DescriptorModel` classes have to manage are:
    * Computation of the descriptor from the element features and the relative position information.
    * Default out feature names. It is the only class that know exactly how many features is able to output.
    * Type of output ['dict', 'ndarray']. It manages the transformation if it is needed in order to output the descriptors in that type.

An example of use within the framework is:
```python
retrievers_list = [ret0]
gret = CollectionRetrievers(retrievers_list)
feats_ret = FeaturesRetriever(features_list, Countdescriptor())
sp_model = SpatialDescriptorModel(gret, desc)
sp_corr = sp_model.compute_nets()
```

## Framework code usage
The framework allows the user to code specific parts of the code, respecting the standards of the framework in order to adapt the code to a specific problem. The main (and prepared) adaptable parts of the code (*hot spots*) are:
* `BaseRetriever` (or even its son classes): where it happens the connection (and possibly the storage) of the spatial data, as well as the retrieve.
* `BaseFeatures` (or even its son classes): where it happens the connection (and possibly the storage) of the feature data, as well as the retrieve and computation of descriptors).
* `BaseDescriptorModel`: where it is computed the descriptors of the selected element from the associated spatial neighborhood retrieved.
* `BasePerturbations`: where it is implemented the statistical random models to apply to the data in order to perform posterior statistical testing.
* `BaseRelativePositioner`: where it is implemented the basic relative position quantity (distance, similarity or other complex object) definition for assign a posteriori measure between elements and the elements in their neighborhoods after the retrieve. 

#### Retriever
Using the `BaseRetriever` coded in the framework, we are able to compute the neighborhood of an element taking into account the retrieve definition we prefer. The framework provides a generalization of the tasks involved in preparing the input indexes, managing the minimal flow needed from the input information on the instantiation and preparing the output in the proper way to be understood for the features data.

In order to use it we have to call it as:
```python
from pySpatialTools.base import BaseRetriever

```

The main features provided by `BaseRetriever` are:
* *iteration* (with `set_iter` and `__iter__`): iterates along all the possible elements retrieving their neighborhood.
* *managing perturbations* (with `add_perturbations` and the retriever functions which deal with it): the perturbation are divided in the ones which affects the retrieving of the neighborhood and the ones which not. If the perturbations follow the standards of the framework, the ones which affects the spatial elements, the class manage to store all the possible neighborhoods in the neighborhood information storage `neighs_info`.
* *manage self-excluding* (with `_format_exclude`), 
* *managing inputs* (with `_prepare_input`): using the information given in the instantiation transforms the inputs into the preferable format of indexes the retriever prefers. In order to know that it uses the parameter `preferable_input_idx`.
* *standarization of output neighborhood information* (with `_format_neighs_info` formatting): taking into account the information given in the instantiation and the perturbations applied, the `_format_neighs_info` format the container `self.neighs_info` variable and the retriever functions taking into account the perturbations applied.
* *compute the network* related with the whole retriever by using `compute_neighnet` function.

The main requirements the `BaseRetriever` needs:
* `__init__`: the instantiation function. It is recommended to follow some order.
* `_format_output` candidates (`_format_output_exclude`, `_format_output_noexclude`), which get the output 
* `_define_retriever`: used in the perturbation functions in order to create parallel perturbed retrievers.
* `_get_loc_from_idx` and `_get_idx_from_loc`: which defines the transformation of indexes to spatial elements and the contrary by interacting with the database.
* `_retrieve_neighs_spec` candidates (`_retrieve_neighs_general_spec`, `_retrieve_neighs_constant_nodistance` and `_retrieve_neighs_constant_distance`) which performs the main retriever function. It defines the interaction with the database. It usually has 3 steps:
    * `_prepare_input`: to automatically prepare the input to the preferable 
    * core retriever: interaction with the DB.
    * `_apply_relative_pos_spec` function which acts as an internal interface with the relative_pos object.
* `_default_ret_val` class parameter: defines the default retriever value.
* `constant_neighs` class parameter: defines if the neighborhood is always with the same number of neighs. It is done to perform an extra optimization in small computations by using numpy.
* `preferable_input_idx` class parameter: the preferable inputs of the core retriever.
* `auto_excluded` class parameter: which informs the class if the core-retriever autoexclude the element from their neighbourhood or no. It is done in order to save time to try to exclude elements that are not needed to exclude.
* `bool_listind` class parameter: if it is possible to get spatial information in a bunch of list of indices or not. It is done in order to take profit when it is possible that possibility and save time.
* overwriting functions of the `BaseRetriever` as the iteration ones.


The general advises and remembers are:
* Keep order in the initialization functions.
* Use the son in inheritance when it is possible and fits your needs. These classes are:
   * `SpaceRetriever`: space-based retriever. Implicit neighborhood, we have to use the functional definition of the neighborhood, and compute the neighborhood on the fly.
   * `NetworkRetriever`: network-based retriever. Explicit computed neighborhood, we only have to retrieve the pre-computed neighborhood.

The main TODOs are:
- [ ] Split the core Retriever in order to isolate properly this part and make easier the coding part for the users.


#### Features
Using the `BaseFeatures` coded in the framework, we are able to retrieve the associated features and the descriptors from them. This class provides to us main tools to get indexes, from them get the features and compute the descriptors. The framework provides a generalization of the tasks involved in preparing the input indexes, managing the minimal flow needed from the input information on the instantiation and preparing the output in the proper way to be understood for the features data.

In order to use it we have to call it as:
```python
from pySpatialTools.base import BaseFeatures

```

The main features provided by `BaseFeatures` are:
* The interaction with different indexing get item information (among them the `pySpatialTools.Neighs_Info`).
* Tools to manage features defined as matrices but also as list of key-values elements.
* Perturbation management tools.

The general advises and remembers are:
* Use the son in inheritance when it is possible and fits your needs. These classes are:
    * `ImplicitFeatures`: the perturbations are computed on the fly.
    * `ExplicitFeatures`: the perturbations are precomputed.
    * `PhantomFeatures`: not features. It uses the descriptormodel directly over the indexes passed.


#### Descriptormodels
Using the `BaseDescriptormodel` coded in the framework, we are able to compute easily the descriptors from the neighborhood features and neighborhood information.  The framework provides a generalization of the tasks involved in computing descriptors and managing the process of interaction with features.

```python
from pySpatialTools.base import BaseDescriptorModel

```

The main features provided by `BaseDescriptorModel` are:
* Possibility to interact with the whole result.
* Predefined interaction with features.

The main requirements to code a correct `BaseDescriptorModel` class is ensuring that they contain:
* `compute` function: the function used to compute the descriptors from the features of neighborhood and the relative positions.
* `name_desc` class parameter: the description of the specific descriptormodel.
* `_nullvalue` class parameter: the null value when there is null neighborhood.


#### Perturbations
Using the `BasePerturbations` coded in the framework, we are able to compute easily the random statistical perturbations in order to perform statistical testing over our measures and models.

In order to use it we have to call it as:
```python
from pySpatialTools.base import BasePerturbations

```

The main requirements to code a correct `BasePerturbations` class is ensuring that they contain:
* `_categorytype` class parameter: which defines if it perturbs over spatial data, features data or both. It could take the respective values of ['general', 'location', 'feature'].
* `_perturbtype` class parameter: which describes the specific method used.


The main TODOs are:
- [x] Code different possible examples.


#### RelativePositioner
The RelativePositioner class is the class which has the task of computing the related position measure of the element with its neighborhood elements. Its object is applied after retrieving the neighborhood of the selected element.

In order to use it we have to call it as:
```python
from pySpatialTools.base import BaseRelativePositioner

```

The main requirements to code a correct `BaseRelativePositioner` class is ensuring that they contain:
* `compute` function, which computes from the spatial element and the spatial elements of the neighborhood.


The main TODOs are:
- [ ] Code different possible examples.

## Examples
The package comes with some examples of use. The examples are selected in two types, taking into account the type of the spatial data:
* *Regular data examples*: the spatial data comes in a regular way, with equidistant and regular defined neighborhoods. This is not the best case to use this package, because it don't allow the user to vectorize, so other orientations could be more satisfactory. The examples which comes with this package are:
    * Natural language processing (`example_nlp.py`): which it is used in order to get descriptors or patterns in texts. There are better tools to study text with python, but with this package we are also able to used in that way. The position in text or the paragraph they belong could be coded as a coordinates. We can also define a neighborhood (and relative positions) using that information in order to retrieve neighbors. We can study the relationship between them (using N-gram model) or more ambitious models using *POS tagging* as a features to try to infer more complex high-level patterns. 
    * Time-series (`example_ts.py`): which it is used to get descriptors or study the time series. The time series measures here are supposed to be sampled equidistant from one to the next. The definition of the neighborhood uses that spatial information. The value of the time-series give us the features data.
    * Images (`example_image.py`): which it is used to get visual descriptors as HOG or SIFT descriptors. Of course, that is not the best tool to get those descriptors neither to study images (there are vectorized and GPU-oriented computing tools for that), but it could be use that problem to understand the package and how it works. The image positions are the spatial information and we can define neighborhood from them. The features are the colors or other mode information we are receiving (gray-scale, RGB, heat, deep, ...).

* *Irregular data examples*: the spatial data comes in a irregular way, with not equal distance or direct regular spatial structure. The examples which comes with this package are:
    * k-means (`example_EM.py`): the n-dimensional data sampled over the space, defined by some statistical patterns could be difficult to cluster, but we could use this package to do it. We can define even local metrics to fit better some patterns if we have this a priori information.
    * Non-homogeneously sampled time series (`example_nhts.py`): in which the data samples are not regular in time. There are ways to put the data regular from the irregular one using interpolation, but others, as the event time series, the only way to do it is losing valuable information. To interpolate or also use local or more problem-adapted techniques we could use this package.
    * Geospatial data (`example_geo.py`): in which the data points are distributed along the geospatial 2-d space. We are going to use testing tools and play with them. Also use 2-d interpolation for qualities homogeneously distributed.




