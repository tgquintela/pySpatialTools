
"""
Mapping tools.

"""


class DummyMap:
    "Dummy map which returns the index input."

    _n = 0

    def __init__(self, n=0):
        self._n = n

    def __getitem__(self, i):
        return i

    def __len__(self):
        return self._n


def create_mapping_valsi(sp_typemodel='', mapper=None):
    bool0 = sp_typemodel == '' or type(sp_typemodel) != str
    if bool0 and mapper is None:
        if sp_typemodel not in ['matrix', 'correlation']:
            raise TypeError("Not correct inputs for defining the mapping.")
    if mapper is not None:
        # Check
        mapper[0]
    else:
        if sp_typemodel == 'correlation':
            raise TypeError("Not correct inputs for defining the mapping.")
        else:
            mapper = DummyMap()
    return mapper
