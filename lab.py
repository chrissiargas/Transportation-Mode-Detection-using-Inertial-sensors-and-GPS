import time

import numpy as np
a = np.array([1,2,2,2,2,3,3,3,4,4])
v = np.where(np.diff(a) != 0)[0] + 1
print(v)
