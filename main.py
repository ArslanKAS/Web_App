

import streamlit as st
# Setting width of App
# st.set_page_config(layout="wide")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import plotly.figure_factory as ff
import seaborn as sns
from p1_eda import load_csv
import p2_manual_eda as p2
import p3_models as p3

# Title Image
st.image("https://i.imgur.com/PpEvWEf.png", caption='Arsalan Ali | arslanchaos@gmail.com', width=800)

# Upload file from PC
with st.sidebar:
    st.subheader("Upload your dataset (.csv)")
    uploaded_file = st.file_uploader("Upload your file", type=["csv"])
    st.info("Awaiting CSV file")

# Navigation Bar
selected = option_menu(
    menu_title=None,
    options=["EDA", "Manual EDA", "Model Prediction", "About us"],
    orientation="horizontal"
)


# Conditions to show Uploaded dataset or use example dataset


# Navigation Bar Mechanism
if selected == "EDA":
    st.subheader(f"Upload dataset or use example for Auto EDA")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        load_csv(df)
    else: 
        if st.button("Press for example data"):
            df = sns.load_dataset("tips")
            load_csv(df)


    # End of EDA Page
#///////////////////////////////////////////////////////////// 
elif selected == "Manual EDA":
    st.subheader(f"Kindly upload the dataset to view options")
        # making columns
    first, second, third = st.columns(3)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        with first:
            st.write(p2.find_shape(df))
            st.write(p2.find_size(df))
            st.write(p2.find_ndim(df))
            st.write(p2.find_describe(df))
        with second:
            st.write(p2.find_unique(df))
            st.write(p2.find_col_nam(df))
            st.write(p2.find_null(df))
        with third:
            p2.custom_plot(df)
    # End of Manual EDA Page
#///////////////////////////////////////////////////////////// 
elif selected == "Model Prediction":
    
    st.subheader(f"Currently prediction on custom datasets not available. Choose from sidebar box")

    # Setting the display box for Dataset on Left Sidebar
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
    # Initializing the function to get the dataset
    X, y = p3.get_dataset(dataset_name)

    # Showing the Shape of X
    st.write("shape of dataset", X.shape)

    # Showing number of unique values/classes in Y
    st.write("number of classes", len(np.unique(y)))

    # Setting the display of ML Model on Left Sidebar
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

    params = p3.add_parameter_ui(classifier_name)

    # Initializing the ML Model function
    clf = p3.get_classifer(classifier_name, params)

    # Classification
    X_train, X_test, y_train, y_test = p3.train_test_split(X, y, test_size=0.2, random_state=20)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = p3.accuracy_score(y_test, y_pred)
    cm = p3.confusion_matrix(y_test, y_pred)

    st.write(f"classifier = {classifier_name}")
    st.write(f"accuracy = {acc}")

     # Plotting Confusion Matrix
    fig2 = plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="viridis")
    plt.ylabel("Actual Output")
    plt.xlabel("Predicted Output")
    all_sample_title = "Accuracy Score: {0}".format(acc)
    plt.title(all_sample_title, size = 8)

    st.pyplot(fig2)

    # End of Prediction Model Page
#///////////////////////////////////////////////////////////// 
elif selected == "About us":
    st.subheader("A team of Codanics ( We all love Pandas )")
    # Dancing pandas
    st.image("https://i.pinimg.com/originals/25/17/39/251739c54e923b2bcc3c89252b6c0e56.gif", caption='dancing panda')
    # st.image("https://i.gifer.com/origin/22/22f37c8f7601e0f4847e99442433c5c4_w200.gif", caption='dancing panda')