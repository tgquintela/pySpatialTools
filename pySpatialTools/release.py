"""Release data for pySpatialTools.

The information of the version is in the version.py file.

"""

from __future__ import absolute_import

import os
import sys
import time
import datetime

basedir = os.path.abspath(os.path.split(__file__)[0])

## Quantify the version
MAJOR = 0
MINOR = 0
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''


def write_version_py(filename=None):
    cnt = """\
version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'pySpatialTools', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (version))
    finally:
        a.close()


def write_versionfile():
    """Creates a static file containing version information."""
    versionfile = os.path.join(basedir, 'version.py')

    text = '''"""
Version information for pySpatialTools, created during installation by
setup.py.

Do not add this file to the repository.

"""

import datetime

version = %(version)r
date = %(date)r

# Development version
dev = %(dev)r

# Format: (name, major, minor, micro, revision)
version_info = %(version_info)r

# Format: a 'datetime.datetime' instance
date_info = %(date_info)r

# Format: (vcs, vcs_tuple)
vcs_info = %(vcs_info)r

'''

    # Try to update all information
    date, date_info, version, version_info, vcs_info = get_info(dynamic=True)

    def writefile():
        fh = open(versionfile, 'w')
        subs = {
            'dev': dev,
            'version': version,
            'version_info': version_info,
            'date': date,
            'date_info': date_info,
            'vcs_info': vcs_info
        }
        fh.write(text % subs)
        fh.close()

    ## Mercurial? Change that
    if vcs_info[0] == 'mercurial':
        # Then, we want to update version.py.
        writefile()
    else:
        if os.path.isfile(versionfile):
            # This is *good*, and the most likely place users will be when
            # running setup.py. We do not want to overwrite version.py.
            # Grab the version so that setup can use it.
            sys.path.insert(0, basedir)
            from version import version
            del sys.path[0]
        else:
            # Then we write a new file.
            writefile()
    return version


def get_revision():
    """Returns revision and vcs information, dynamically obtained."""
    vcs, revision, tag = None, None, None

    hgdir = os.path.join(basedir, '..', '.hg')
    gitdir = os.path.join(basedir, '..', '.git')

    if os.path.isdir(gitdir):
        vcs = 'git'
        # For now, we are not bothering with revision and tag.

    vcs_info = (vcs, (revision, tag))

    return revision, vcs_info


def get_info(dynamic=True):
    ## Date information
    date_info = datetime.datetime.now()
    date = time.asctime(date_info.timetuple())

    revision, version, version_info, vcs_info = None, None, None, None

    import_failed = False
    dynamic_failed = False

    if dynamic:
        revision, vcs_info = get_revision()
        if revision is None:
            dynamic_failed = True

    if dynamic_failed or not dynamic:
        # All info should come from version.py. If it does not exist, then
        # no vcs information will be provided.
        sys.path.insert(0, basedir)
        try:
            from version import date, date_info, version, version_info,\
                vcs_info
        except ImportError:
            import_failed = True
            vcs_info = (None, (None, None))
        else:
            revision = vcs_info[1][0]
        del sys.path[0]

    if import_failed or (dynamic and not dynamic_failed):
        # We are here if:
        #   we failed to determine static versioning info, or
        #   we successfully obtained dynamic revision info
        version = ''.join([str(major), '.', str(minor), '.', str(micro)])
        if dev:
            version += '.dev_' + date_info.strftime("%Y%m%d%H%M%S")
        version_info = (name, major, minor, micro, revision)

    return date, date_info, version, version_info, vcs_info


## Version information
name = 'pySpatialTools'
major = "0"
minor = "0"
micro = "0"
## Declare current release as a development release.
## Change to False before tagging a release; then change back.
dev = True

description = """Python package for studying spatial irregular heterogenous
data."""
long_description = """
This package is built in order to provide prototyping tools in python to deal
with spatial data in python and model spatial-derived relations between
different elements in a system. In some systems, due to the huge amount of
data, the complexity of their topology their local nature or because other
practical reasons we are forced to use only local information for model the
system properties and dynamics.

pySpatialTools is useful for complex topological systems with different type
of spatial data elements and feature data elements in which we are not able to
study alls at once because of the data size.

pySpatialTools could be not recommendable for treating some specific problems
with homogeneous and/or regular data which could be treated with other python
packages, as for example computational linguistics (nltk), computer vision or
grid data (scipy.ndimage and openCV) or others.

"""

## Main author
author = 'T. Gonzalez Quintela',
author_email = 'tgq.spm@gmail.com',

license = 'MIT'
authors = {'tgquintela': ('T. Gonzalez Quintela', 'tgq.spm@gmail.com')}

maintainer = ""
maintainer_email = ""

url = ''
download_url = ''

platforms = ['Linux', 'Mac OSX', 'Windows', 'Unix']

keywords = ['math', 'data analysis', 'Mathematics', 'spatial networks',
            'spatial correlations', 'framework', 'social sciences',
            'spatial analysis', 'spatial ecology']

classifiers = [
    #  How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',
    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    # Specify the Python versions you support here
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    # Topic information
    'Topic :: Software Development :: Build Tools',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Scientific/Engineering :: Sociology',
    'Topic :: Scientific/Engineering :: Data Analysis',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Mathematics']

date, date_info, version, version_info, vcs_info = get_info()

if __name__ == '__main__':
    # Write versionfile for nightly snapshots.
    write_versionfile()
