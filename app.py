import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import gdown

# Title and description
st.title("Balls Classification App")
st.write("Classes: Shuttlecock, Volleyball, Tennis Ball, Basketball, Baseball")

# Map the class number to the ball class name
cat_to_name = {
    '0': 'Baseball', 
    '1': 'Basketball', 
    '2': 'Shuttlecock', 
    '3': 'Tennis Ball', 
    '4': 'Volleyball'
}

# Load the pre-trained VGG model with modification for 5 classes (balls)
vgg_model = models.vgg11(pretrained=True)
vgg_model.classifier[6] = torch.nn.Linear(vgg_model.classifier[6].in_features, 5)

# Try to load the model, download if not available
try:
    vgg_model.load_state_dict(torch.load('./balls_classification.pth', map_location=torch.device('cpu')))
except FileNotFoundError:
    with st.spinner("Downloading model. Please wait..."):
        # Download model from Google Drive (make sure this is the correct link for the ball model)
        url = 'https://drive.google.com/uc?id=1tAl7mNIfGPe2ulyzv3CGlOmI_Ft26WeF'  # Change link if needed
        output = './balls_classification.pth'
        gdown.download(url, output, quiet=False)
        vgg_model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))

vgg_model.eval()

st.write("Please upload an image of a ball for the model to make predictions.")

# File uploader for image
image = st.file_uploader("Upload an image (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if image is not None:
    # Display the uploaded image
    st.image(image, caption="Uploaded image.", use_column_width=True)

    # Preprocess the image
    img = Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = vgg_model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()

    # Display the result
    st.write(f"Predicted class: {cat_to_name[str(predicted_class)]}")
