# FILTERS OUT DUPLICATE QUADHASHES FROM THE NLCD OUTPUT

import csv
import scipy.spatial as spatial
row_maps = {}
vector1 = [1, 2, 3]
vector2 = [3, 2, 1]

cosine_similarity = 1 - spatial.distance.cosine(vector1, vector2)
print(cosine_similarity)