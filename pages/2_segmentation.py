import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("SEM Image Segmentation (Highest Area Fraction)")

st.markdown("""
Upload an SEM image. The app will:
- Apply Otsu thresholding,
- Show both possible segmentations,
- Display the one with the highest area fraction (white pixels),
- Show the area fraction value.
""")

uploaded_file = st.file_uploader("Upload SEM Image", type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('L'))  # Convert to grayscale

    # Otsu thresholding
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask1 = binary > 0
    mask2 = ~mask1
    area_fraction1 = np.mean(mask1)
    area_fraction2 = np.mean(mask2)

    if area_fraction1 >= area_fraction2:
        best_mask = mask1
        best_fraction = area_fraction1
    else:
        best_mask = mask2
        best_fraction = area_fraction2

    st.subheader("Original Image")
    st.image(img, use_container_width=True, clamp=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Segmentation 1 (area fraction: {:.2f})**".format(area_fraction1))
        st.image(mask1.astype(np.uint8)*255, use_container_width=True, clamp=True)
    with col2:
        st.write("**Segmentation 2 (area fraction: {:.2f})**".format(area_fraction2))
        st.image(mask2.astype(np.uint8)*255, use_container_width=True, clamp=True)

    st.subheader("Best Segmentation (Highest Area Fraction: {:.2f})".format(best_fraction))
    st.image(best_mask.astype(np.uint8)*255, use_container_width=True, clamp=True)

    # Download option
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray((best_mask.astype(np.uint8)*255)).save(buf, format="TIFF")
    st.download_button(
        label="Download Best Segmentation",
        data=buf.getvalue(),
        file_name=f"segmented_{uploaded_file.name}",
        mime="image/tiff"
    )
else:
    st.info("👆 Upload an SEM image to get started.")
