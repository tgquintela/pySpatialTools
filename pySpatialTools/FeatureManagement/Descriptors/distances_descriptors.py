
"""
DistancesDescriptor
-------------------
This is a dummy descriptor for a phantom features. It uses the spatial relative
position of the neighbourhood information to compute descriptors.

"""



class Countdescriptor(DescriptorModel):
    """Model of spatial descriptor computing by counting the type of the
    neighs represented in feat_arr.

    """
    name_desc = "Counting descriptor"
    _nullvalue = 0

    def __init__(self, funct=None, type_infeatures=None, type_outfeatures=None):
        """The inputs are the needed to compute model_dim."""
        ## Initial function set
        self._format_default_functions()
        self.set_functions(type_infeatures, type_outfeatures)
        ## Check descriptormodel
        self._checker_descriptormodel()


def distances_descriptors(pointfeats, point_pos, f):
    """Distances descriptors.
    """
    descriptors = []
    for k in range(len(pointfeats)):
        descriptors_k = []
        for i in range(len(pointfeats[k])):
            descriptors_ki = {}
            for nei in range(len(pointfeats[k][i])):
                descriptors_ki[pointfeats[k][i][nei]] = f(point_pos[k][i][nei])
            descriptors_k.append(descriptors_ki)
        descriptors.append(descriptors_k)
    return descriptors
