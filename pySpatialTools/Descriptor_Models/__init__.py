
"""
Models
======
Package which contains the classes needed to compute matrix descritptors or
spatial correlation matrices.
"""

from descriptor_models import DescriptorModel
from model_process import ModelProcess

## Descriptor models
from pjensen import Pjensen
from countdescriptors import Countdescriptor