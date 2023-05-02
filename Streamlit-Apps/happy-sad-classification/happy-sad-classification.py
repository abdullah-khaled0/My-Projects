import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model('Streamlit-Apps/happy-sad-classification/imageclassifier.h5')

# Define the Streamlit app
def app():
    # Set the app title
    st.title('Happy😄 or Sad🥺')

    # Add a file uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Check if an image has been uploaded
    if uploaded_file is not None:
        # Read the image data
        image = Image.open(uploaded_file)

        # Resize the image to match the expected input shape of the model
        image = image.resize((256, 256))

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Normalize the pixel values
        image_array = image_array / 255.0

        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Make predictions on the image
        predictions = model.predict(image_array)

        if predictions >= 0.5: 
            predicted_class_name = 'sad🥺'
        else:
            predicted_class_name = 'happy😄'

        # Display the image and the predicted class name
        st.image(image, caption='', use_column_width=True)

        # Add a separator
        st.write('---')

        # Display the predicted class name
        st.markdown(f'## {predicted_class_name.capitalize()}')
# Run the app
if __name__ == "__main__":
    app()
