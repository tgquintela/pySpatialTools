
## Tested properly

import numpy as np
import pandas as pd

from pySpatialTools.Preprocess import Aggregator
from pySpatialTools.IO import create_syntetic_data, create_reindices

n, m, m_agg, m_feat = 100000, 10, 100, 20
df, typevars = create_syntetic_data(n, m_agg, m_feat)
reindices = create_reindices(n, m)

agg = Aggregator(typevars=typevars)

res1, res2 = agg.retrieve_aggregation(df, reindices)

