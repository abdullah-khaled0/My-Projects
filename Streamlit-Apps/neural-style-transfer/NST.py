import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import io

from PIL import Image

@st.cache_resource
def load_model():
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return model

def load_image(img_data):
    img = tf.image.decode_image(img_data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def gen_image():
    st.title("Generative AI - Neural Style Transfer")

    img1 = st.file_uploader("Content Image", type=["jpg", "jpeg", "png"])
    img2 = st.file_uploader("Style Image", type=["jpg", "jpeg", "png"])

    if img1 and img2:
        # Read image data
        img1_data = img1.read()
        img2_data = img2.read()

        # Open images
        image1 = Image.open(io.BytesIO(img1_data))
        image2 = Image.open(io.BytesIO(img2_data))

        # Resize images to have the same width and height
        size = (200, 200)
        image1 = image1.resize(size)
        image2 = image2.resize(size)

        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, use_column_width=True, caption="Content Image")
        with col2:
            st.image(image2, use_column_width=True, caption="Style Image")

        # Button to generate stylized image
        if st.button("Generate"):
            content_image = load_image(img1_data)
            style_image = load_image(img2_data)
            model = load_model()

            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

            # Convert stylized image to PIL format for display
            stylized_image = tf.squeeze(stylized_image).numpy()
            stylized_image = np.clip(stylized_image, 0, 1)
            stylized_image = (stylized_image * 255).astype(np.uint8)
            stylized_image = Image.fromarray(stylized_image)

            # Display the stylized image
            st.image(stylized_image, use_column_width=True, caption="Stylized Image")

gen_image()
