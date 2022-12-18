import streamlit as st
import pandas as pd
import numpy as np

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)



x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

st.bokeh_chart(chart_data)
