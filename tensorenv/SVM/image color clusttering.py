import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
from sklearn.preprocessing import scale
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt

n_colors = 64

# Load the Summer Palace photo
china = datasets.load_sample_image("china.jpg")
# print(china)

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(china, dtype=np.float64) / 255
# print(china)

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
print(w,h,d)
print(original_shape)
# assert d == 3
print(w,h,d)
image_array = np.reshape(china, (w * h, d))
print(len(image_array))