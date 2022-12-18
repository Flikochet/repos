import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
@st.experimental_memo
arr = np.random.normal(1, 1, size=100)
fig, ax = px.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)
