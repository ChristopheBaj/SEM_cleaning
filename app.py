import streamlit as st
import os

os.environ(['OPENCV_IO_ENABLE_OPENEXR'])=0
os.environ(['OPENCV_IO_ENABLE_JASPER'])=0

st.set_page_config(
    page_title="SEM Data Cleaning & Segmentation",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 SEM Data Cleaning & Segmentation App")

st.markdown("""
Welcome to the SEM Data Processing App!

**What can you do here?**
- **Remove vertical stripes and artifacts** from your SEM images (see "SEM Cleaning" page).
- **Segment your images** and display the segmentation with the highest area fraction (see "Segmentation" page).

---

### **How to use this app**
1. Use the sidebar to navigate between:
    - **SEM Cleaning**: Clean your SEM images using median and FFT-based methods.
    - **Segmentation**: Segment your images and view the segmentation with the highest area fraction.
2. Upload your SEM images (supports `.tif`, `.png`, `.jpg`, etc.).
3. Adjust parameters and download your results.
""")

st.markdown("""
---

**Navigate using the sidebar to get started!**
""")
