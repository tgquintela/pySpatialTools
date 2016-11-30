
# TODO
General list of TODOs.

## Technical TODO list
Technical improvements TODO:
- [ ] **Warning numpy correction** */usr/local/lib/python2.7/dist-packages/numpy-1.11.0rc1-py2.7-linux-x86_64.egg/numpy/core/fromnumeric.py:2652: VisibleDeprecationWarning: `rank` is deprecated; use the `ndim` attribute or function instead. To find the rank of a matrix see `numpy.linalg.matrix_rank`.
  VisibleDeprecationWarning)*
- [ ] **Cover some unnecessary warnings**
- [ ] **Uniform all the features inputs**: nullvalue and indices
- [ ] **Coordination between add2result and initialization in resulter**: sometimes it is created a matrix initialization with `append_addresult_function`. It is dependent on the `map_vals_i.n_in` and `map_vals_i.n_out`.
- [ ] **`get_loc_i` list of indices fail to retrieve**
- [ ] **Connection with QGIS**
- [ ] **Support for multiple indexes retrieve**
- [ ] **Spatial hashing into the discretization module**.

## Functionality TODO list
Improvements in the functionality TODO:
- [ ] **To_complete_result**: use the descriptormodel one by default, if it is not given in the resulter.
- [ ] **Bad windows retriever results**: It excludes borders and corners.
- [ ] **Support index and multi-index in features**
- [ ] **Spatially constraint sampling**
- [ ] **Interface for changing neighbourhood*: for exploring best or optimal neighbourhood parameters. In a way it can be used as a convolutional layer of an ANN.
- [ ] **Permutations ration class**

## Usability TODO list
Improvements in the usability TODO:
- [ ] **api module**: in which is collected all the basic common uses of the package.
- [ ] **statistical tests interface**
