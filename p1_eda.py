import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# @st.cache(suppress_st_warning=True) 
def load_csv(df):
    pr = ProfileReport(df, explorative=True)
    st.subheader("**Input Dataframe**")
    st.write(df)
    st.write("---")
    st.subheader("**Profiling report with Pandas**")
    st_profile_report(pr)