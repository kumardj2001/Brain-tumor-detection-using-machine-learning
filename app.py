import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same model architecture
class BrainTumorModel(torch.nn.Module):
    def __init__(self):      
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 256, kernel_size=3),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(256, 32, kernel_size=2)
        )
        self.linear1 = torch.nn.Linear(62, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.flat = torch.nn.Flatten(1)
        self.linear3 = torch.nn.Linear(126976, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.flat(x)
        x = self.linear3(x)
        return x

# Load model
model = BrainTumorModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image (grayscale) to check for brain tumor.")

uploaded_file = st.file_uploader("Upload MRI image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert("L").resize((128, 128))  # Grayscale and resize
    array = np.asarray(image).reshape(1, 1, 128, 128)
    tensor = torch.tensor(array, dtype=torch.float32).to(device)
    return tensor

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(tensor)
            prediction = torch.argmax(outputs, dim=1).item()

        label_map = {0: "🟢 No Tumor Detected", 1: "🔴 Tumor Detected"}
        st.markdown(f"### Result: {label_map[prediction]}")
