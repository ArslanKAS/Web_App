import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

def find_shape(df):
    if st.button("Find out the shape"):
     return df.shape

def find_size(df):
    if st.button("Find out the size"):
     return df.size

def find_ndim(df):
    if st.button("Find out the dimensions"):
     return df.ndim

def find_describe(df):
    if st.button("Find out the summary statisitcs"):
        return df.describe()

def find_unique(df):
    if st.button("Find out number of unique values"):
        return df.nunique()

def find_col_nam(df):
    if st.button("Find out column names"):
        return df.columns

def find_null(df):
    if st.button("Find out number of null values"):
        return df.isnull().sum()

def custom_plot(df):
    column_names = df.columns
    col_options = st.multiselect('Select the features',column_names)
    col_obj = np.ravel(col_options).astype("object")
    st.write("To be continued")
    # st.write(np.ravel(col_options).astype("object"))
    # if col_options is not None:
        # fig = ff.create_distplot(df[np.ravel(col_options)], column_names) #bin_size=[.1, .25, .5]
    # # if st.button("Plot the histogram"):
    #     st.plotly_chart(fig, use_container_width=True)