import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage import measure
import pandas as pd
import matplotlib.pyplot as plt

st.title("SEM Image Segmentation & Region Measurements")

st.markdown("""
Upload an SEM image. The app will:
- Apply Otsu thresholding and select the segmentation with the highest area fraction
- Display the best segmentation
- Measure region properties (area, perimeter, eccentricity, centroid, etc.)
- Show and allow download of the measurements table
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

    st.subheader(f"Best Segmentation (Highest Area Fraction: {best_fraction:.2f})")
    # Show results
    figure = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Segmented (Otsu) Black = Matter, White = Space')
    plt.imshow(best_mask.astype(np.uint8)*255, cmap='gray')
    plt.axis('off')
    st.pyplot(figure)

    # --- Region Measurements ---
    labels = measure.label(best_mask)
    props = measure.regionprops_table(
        labels,
        properties=[
            'area', 'perimeter', 'eccentricity', 'centroid',
            'orientation', 'major_axis_length', 'minor_axis_length'
        ]
    )
    features_df = pd.DataFrame(props)
    features_df['area_fraction'] = np.mean(best_mask)

    st.subheader("Region Measurements Table")
    st.dataframe(features_df)

    # Download option
    csv = features_df.to_csv(index=False).encode()
    st.download_button(
        label="Download Measurements as CSV",
        data=csv,
        file_name=f"segmentation_measurements_{uploaded_file.name}.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload an SEM image to get started.")
