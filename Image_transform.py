import streamlit as st
import cv2
import numpy as np

st.title("Image Affine Transformations")

image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image is not None:
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]

    st.image(image, caption="Uploaded Image", use_column_width=True)

    translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])

    angle = st.slider("Rotation Angle (degrees)", -180, 180, 0)
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    scale_x = st.slider("Scale X", 0.1, 3.0, 1.0)
    scale_y = st.slider("Scale Y", 0.1, 3.0, 1.0)
    scaling_matrix = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])

    shear_x = st.slider("Shear X", -1.0, 1.0, 0.0)
    shear_y = st.slider("Shear Y", -1.0, 1.0, 0.0)
    shear_matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])

    if st.button("Apply Transformations"):
        translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        scaled_image = cv2.warpAffine(image, scaling_matrix, (width, height))
        sheared_image = cv2.warpAffine(image, shear_matrix, (width, height))

        st.image(translated_image, caption="Translated Image", use_column_width=True)
        st.image(rotated_image, caption="Rotated Image", use_column_width=True)
        st.image(scaled_image, caption="Scaled Image", use_column_width=True)
        st.image(sheared_image, caption="Sheared Image", use_column_width=True)
