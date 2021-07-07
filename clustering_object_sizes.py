import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as sm


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

object_sizes = pd.read_csv("object_sizes.csv")

# print(object_sizes)

# plt.scatter(x=object_sizes["width"], y=object_sizes["height"])

X = np.array(object_sizes[["width", "height"]]).reshape(-1, 2)

print("K-means:")

kmeans_model = KMeans(n_clusters=5)

kmeans_model.fit(X)

kmeans_classes = kmeans_model.predict(X)

kmeans_db_score = sm.davies_bouldin_score(X, kmeans_classes)
print("Davies-Bouldin score: {0} (less is better)".format(kmeans_db_score))

ax1.set_title("K-means")
ax1.scatter(x=object_sizes["width"], y=object_sizes["height"], c=kmeans_classes, cmap="prism")

print("GMM:")

gmm_model = GaussianMixture(n_components=5)

gmm_model.fit(X)

gmm_classes = gmm_model.predict(X)

gmm_db_score = sm.davies_bouldin_score(X, gmm_classes)
print("Davies-Bouldin score: {0} (less is better)".format(gmm_db_score))

ax2.set_title("GMM")
ax2.scatter(x=object_sizes["width"], y=object_sizes["height"], c=gmm_classes, cmap="prism")

plt.show()
