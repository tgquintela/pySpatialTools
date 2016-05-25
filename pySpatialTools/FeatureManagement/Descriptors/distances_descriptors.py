
"""
DistancesDescriptor
-------------------
This is a dummy descriptor for a phantom features. It uses the spatial relative
position of the neighbourhood information to compute descriptors.

"""


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
