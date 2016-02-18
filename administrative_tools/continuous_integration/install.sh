#!/bin/bash
# Based on a script from scikit-learn

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
#export CC=gcc
#export CXX=g++

if [[ "$DISTRIB" == "conda_min" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate
    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    export PATH=/home/travis/miniconda/bin:$PATH
    conda config --set always_yes yes
    conda update --yes conda
    conda config --add channels soft-matter
    # Configure the conda environment and put it in the path using the
    # provided versions
    #conda create -n testenv --yes python=$PYTHON_VERSION pip nose coverage six \
    #    numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    conda create -n testenv --yes $DEPS python=$TRAVIS_PYTHON_VERSION
    source activate testenv
    conda install --file administrative_tools/continuous_integration/requirements.txt -y
    #conda install libgfortran

  # for debugging...
    echo $PATH
    which python
    conda info
    conda list

elif [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    export PATH=/home/travis/miniconda/bin:$PATH
    conda config --set always_yes yes
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes $DEPS python=$TRAVIS_PYTHON_VERSION
    source activate testenv
    conda install --file administrative_tools/continuous_integration/requirements.txt -y
  # for debugging...
    echo $PATH
    which python
    conda info
    conda list

    if [[ "$COVERAGE" == "true" ]]; then
        pip install coveralls
    fi

    #python -c "import pandas; import os; assert os.getenv('PANDAS_VERSION') == pandas.__version__"

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    deactivate
    # Create a new virtualenv using system site packages for numpy and scipy
    virtualenv --system-site-packages testenv
    source testenv/bin/activate
    pip install -r administrative_tools/continuous_integration/requirements.txt
    #pip install nose
    #pip install coverage
    #pip install numpy==$NUMPY_VERSION
    #pip install scipy==$SCIPY_VERSION
    #pip install six
    #pip install quantities
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coveralls
fi

## Example of installing personal required packages
#wget https://github.com/../archive/snapshot-code.tar.gz
#tar -xzvf snapshot-code.tar.gz
#pushd package-code-name
#python setup.py install
#popd
python setup.py install

pip install .
