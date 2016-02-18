#!/bin/bash
# Installing
sudo pip install -r requirements.txt
sudo python setup.py install
# Deleting trash files
sudo rm -r build/
sudo rm -r dist/
sudo rm -r pySpatialTools.egg-info/
