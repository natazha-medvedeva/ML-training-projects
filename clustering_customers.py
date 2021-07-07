import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


def set_printing_options():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)


set_printing_options()

clients = pd.read_csv("customer_online_closing_store.csv")
clients["return_rate"] = clients["items_returned"] / clients["items_purchased"]
clients["average_price"] = clients["total_spent"] / clients["items_purchased"]

print(clients[["average_price", "return_rate", "overall_rating"]])

X = np.array(clients[["average_price", "return_rate", "overall_rating"]]).reshape(-1, 3)
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

print(X)

plt.title("Customer dendrogram")
dend = shc.dendrogram(shc.linkage(X, method="single"))

agglomerative_model = AgglomerativeClustering(n_clusters=4, linkage="single")

agglomerative_model.fit(X)

classes = agglomerative_model.labels_

clients["class"] = classes

print(clients[["average_price", "return_rate", "overall_rating", "class"]])

client_pivot_table = clients.pivot_table(index="class",
                                         values=["average_price", "return_rate", "overall_rating", "customer_id"],
                                         aggfunc={"average_price": np.mean, "return_rate": np.mean,
                                                  "overall_rating": np.mean, "customer_id": len})

print(client_pivot_table)

plt.show()
