import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import gdown

st.title("Balls Classification App")
st.write("Classes: Baseball, Basketball, Shuttlecock, Tennis Ball, Volleyball")

# Map the class number to the class name
cat_to_name = {
    '0': 'Baseball', 
    '1': 'Basketball', 
    '2': 'Shuttlecock', 
    '3': 'Tennis_Ball', 
    '4': 'Volleyball'}

# Load the pre-trained VGG model with modification
vgg_model = models.vgg11(pretrained=True)
vgg_model.classifier[6] = torch.nn.Linear(vgg_model.classifier[6].in_features, 5)

try:
    vgg_model.load_state_dict(torch.load('./balls_classification.pth', map_location=torch.device('cpu')))
except FileNotFoundError:
    with st.spinner("Downloading model. Please wait..."):
        # if not found, download state from Google Drive link
        url = 'https://drive.google.com/drive/folders/1bjClzBzOfrVman8Bb16o3SkXcii1chLQ'
        output = './balls_classification.pth'
        gdown.download(url, output, quiet=False)
        vgg_model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))

vgg_model.eval()

st.write("""Please upload an image file for the model to make predictions. The predicted fruit class will be displayed.""")

# File uploader for image
image = st.file_uploader("Upload an image (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if image is not None:
    # Display the uploaded image
    st.image(image, caption="Uploaded image.", use_column_width=True)
    st.write("")

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
