import streamlit as st
import cv2
import numpy as np

def adjust_contrast_brightness(image, alpha, beta):
    height, width, channels = image.shape
    new_image = np.zeros((height, width, channels), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

    return new_image

def apply_gaussian_blur(image, kernel_size):
    height, width, channels = image.shape
    new_image = np.zeros((height, width, channels), dtype=np.uint8)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            for c in range(channels):
                new_image[y, x, c] = np.sum(image[y - 1:y + 2, x - 1:x + 2, c] * kernel)

    return new_image

def apply_sharpening(image):
    height, width, channels = image.shape
    new_image = np.zeros((height, width, channels), dtype=np.uint8)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            for c in range(channels):
                new_image[y, x, c] = np.sum(image[y - 1:y + 2, x - 1:x + 2, c] * kernel)

    return new_image

def create_mask(image, rect):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(rect[0][1], rect[1][1]):
        for x in range(rect[0][0], rect[1][0]):
            mask[y, x] = 255

    return mask

def apply_morphological_operation(mask, iterations):
    height, width = mask.shape
    new_mask = np.zeros((height, width), dtype=np.uint8)
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    for _ in range(iterations):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                new_mask[y, x] = np.max(mask[y - 1:y + 2, x - 1:x + 2] * kernel)

    return new_mask

def apply_mask(image, mask):
    height, width, channels = image.shape
    masked_image = np.zeros((height, width, channels), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                masked_image[y, x, c] = image[y, x, c] * (mask[y, x] / 255)

# Streamlit App
st.title("Image Enhancement App")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Choose an operation
    operation = st.selectbox("Select Operation", ["Original", "Contrast & Brightness", "Gaussian Blur", "Sharpening", "Masking & Morphological"])

    if operation == "Original":
        st.image(image, channels="BGR")
    elif operation == "Contrast & Brightness":
        alpha = st.slider("Contrast (Alpha)", 0.1, 5.0, 1.0)
        beta = st.slider("Brightness (Beta)", -100, 100, 0)
        enhanced_image = adjust_contrast_brightness(image, alpha, beta)
        st.image(enhanced_image, channels="BGR")
    elif operation == "Gaussian Blur":
        kernel_size = st.slider("Kernel Size", 1, 11, 3)
        blurred_image = apply_gaussian_blur(image, kernel_size)
        st.image(blurred_image, channels="BGR")
    elif operation == "Sharpening":
        sharpened_image = apply_sharpening(image)
        st.image(sharpened_image, channels="BGR")
    elif operation == "Masking & Morphological":
        rect_start_x = st.slider("Mask Start X", 0, image.shape[1] - 1, 0)
        rect_start_y = st.slider("Mask Start Y", 0, image.shape[0] - 1, 0)
        rect_end_x = st.slider("Mask End X", 0, image.shape[1] - 1, image.shape[1] - 1)
        rect_end_y = st.slider("Mask End Y", 0, image.shape[0] - 1, image.shape[0] - 1)

        rect = ((rect_start_x, rect_start_y), (rect_end_x, rect_end_y))
        mask = create_mask(image, rect)

        morph_iterations = st.slider("Morphological Iterations", 0, 10, 2)
        mask_dilated = apply_morphological_operation(mask, morph_iterations)

        masked_image = apply_mask(image, mask_dilated)
        st.image(masked_image, channels="BGR")
