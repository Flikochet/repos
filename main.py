import plotly.express as px
import numpy as np

arr = np.random.normal(1, 1, size=100)
fig, ax = px.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)
