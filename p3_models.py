import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
import main as m

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

# Initializing the function to get the dataset
# X, y = get_dataset(m.dataset_name)

# Function to set Parameters of the ML Models
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        # K is the number of neighbours
        K = st.sidebar.slider("K", 1, 15)
        # Adding K key and value to dictionary params
        params["K"] = K
    elif clf_name == "SVM":
        # C is used to control the error during classification
        C = st.sidebar.slider("C", 0.01, 10.0)
        # Adding C key and value to dictionary params
        params["C"] = C
    else:
        # Max depth is the depth of the trees (nodes within nodes)
        max_depth = st.sidebar.slider("max_depth", 2, 20)
        # N_estimator is the number of trees
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        # Adding Max depth key and value to dictionary params
        params["max_depth"] = max_depth
        # Adding N_estimator key and value to dictionary params
        params["n_estimators"] = n_estimators
    return params



# Function to select the ML Model
def get_classifer(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)
    return clf

# # Plotting 
# pca = PCA(2)
# X_projected = pca.fit_transform(X)

# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]

# fig = plt.figure()
# plt.scatter(x1, x2, c=y, alpha=0.8, cmap="rocket_r")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.colorbar()

# st.pyplot(fig)