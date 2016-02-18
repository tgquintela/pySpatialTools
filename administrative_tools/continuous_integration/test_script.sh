#!/bin/bash
# Based on a script from scikit-learn

# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

set -e

if [[ "$COVERAGE" == "true" ]]; then
    nosetests --with-coverage --cover-package=pySpatialTools
else
    nosetests
fi
