import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import torch
from GeneralStripeRemover import GeneralStripeRemover

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="SEM Stripe Removal (Median + GSR)", layout="wide")
st.title("🔬 SEM Stripe Removal: Column-wise Median + GSR")

st.markdown("""
Upload an SEM image. This app will:
1. Remove vertical stripes using column-wise median subtraction
2. Further clean using General Stripe Remover (GSR)
3. Show results and allow download
""")

uploaded_file = st.file_uploader("Upload SEM Image", type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])
col1, col2 = st.columns(2)

if uploaded_file:
    with col1:
        # Load and display original
        image = Image.open(uploaded_file)
        img = np.array(image.convert('L'))  # grayscale
        st.subheader("Original Image")
        st.image(img, use_container_width=True, clamp=True)

    with col2:
        # --- Column-wise Median Stripe Removal ---
        col_median = np.median(img, axis=0)
        stripe_profile = np.tile(col_median, (img.shape[0], 1))
        img_destriped = img.astype(np.float32) - stripe_profile
        img_destriped = cv2.normalize(img_destriped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        st.subheader("After Column-wise Median Stripe Removal")
        st.image(img_destriped, use_container_width=True, clamp=True)

    # --- GSR Stripe Removal ---
    st.subheader("After GSR Stripe Removal")
    img_torch = torch.from_numpy(img_destriped).float()
    img_norm = (img_torch - img_torch.min()) / (img_torch.max() - img_torch.min())

    # GSR parameters (customize as needed)
    iterations = st.number_input("GSR Iterations", min_value=100, max_value=10000, value=2500, step=100)
    mu_main = st.number_input("GSR mu[0]", min_value=0.0, max_value=100.0, value=10.0)
    mu_aux = st.number_input("GSR mu[1]", min_value=0.0, max_value=10.0, value=0.1)
    direction = st.selectbox("Stripe Direction", options=["Vertical (1,0,0)", "Horizontal (0,1,0)"], index=0)
    direction_vec = [1.,0.,0.] if direction.startswith("Vertical") else [0.,1.,0.]

    if st.button("Run GSR"):
        with st.spinner("Running GSR..."):
            result = GeneralStripeRemover(
                img_norm, 
                iterations=int(iterations), 
                proj=True, 
                mu=[mu_main, mu_aux], 
                resz=0, 
                direction=direction_vec, 
                verbose=False
            )
            result_np = result.cpu().numpy()
            result_uint8 = (255 * (result_np - result_np.min()) / (result_np.max() - result_np.min())).astype(np.uint8)
            st.image(result_uint8, caption="GSR Cleaned", use_container_width=True, clamp=True)

            # Download button
            buf = io.BytesIO()
            Image.fromarray(result_uint8).save(buf, format="TIFF")
            st.download_button(
                label="Download Cleaned Image",
                data=buf.getvalue(),
                file_name=f"cleaned_{uploaded_file.name}",
                mime="image/tiff"
            )
        with col1:
            st.subheader("Original Image")
            st.image(img, use_container_width=True, clamp=True)
        with col2:
            st.subheader("Cleaned Image")
            st.image(result_uint8, caption="GSR Cleaned", use_container_width=True, clamp=True)
            
            

else:
    st.info("👆 Upload an SEM image to get started.")

st.markdown("---")
st.markdown("**Tip:** The GSR step is slow for large images. Try cropping or downsampling for testing.")
