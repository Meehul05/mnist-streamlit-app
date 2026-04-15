import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Load model
model = ANN()
model.load_state_dict(torch.load("ann_mnist1.pth", map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("MNIST Digit Classifier")

file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, width=150)

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs).item()

    st.subheader(f"Prediction: {pred}")

    st.write("Confidence:")
    for i, p in enumerate(probs[0]):
        st.write(f"{i}: {p.item():.4f}")

    plt.bar(range(10), probs[0].numpy())
    st.pyplot(plt)