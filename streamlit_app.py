import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import joblib
from st_circular_progress import CircularProgress

# Set title of the app
st.title('Fruit Classifier using CNN')
st.write("Fruits to classify: Acerolas, Apples, Apricots, Avocados, Bananas")

# File uploader for image (accept jpg, png, jpeg)
uploaded_file = st.file_uploader("Upload a fruit image (jpg, jpeg, png)", type=['jpg', 'jpeg', 'png'])


# Load the CNN model using st.cache_resource
@st.cache_resource
def load_model():
    model = joblib.load(r'balls_classification.pkl')
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image).unsqueeze(0) # for batch dimension
    return image_tensor

# Make prediction
def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item(), probabilities

# If an image file is uploaded, display it and make a prediction
if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Convert the image to RGB
    image = image.convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=False)

    # Preprocess the image
    image_tensor = preprocess_image(image)

    class_names = ['Baseball', 'Basketball', 'Shuttlecock', 'Tennis_Ball', 'Volleyball']

    # Predict button
    if st.button('Predict!'):
        predicted_class, probabilities = predict(image_tensor)

        st.subheader("Predictions:")
        
        # Create columns for displaying progress
        cols = st.columns(len(class_names))

        for i, (class_name, probability) in enumerate(zip(class_names, probabilities[0])):

            with cols[i]:  # Use the current column context
                # Convert probability to percentage and format to 2 decimal places
                percentge = probability.item() * 100

                # Update CircularProgress with the correct formatted probability
                my_circular_progress = CircularProgress(
                    label=class_name,
                    value=int(percentge),  # Use formatted probability value
                    key=f"progress_{class_name}_{uploaded_file.name}"  # Ensure unique key for each progress bar based on uploaded file
                )

                my_circular_progress.st_circular_progress()

        st.subheader(f"Most likely class: {class_names[predicted_class]}")