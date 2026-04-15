import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# ------------------ MODEL ------------------
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

# ------------------ LOAD MODEL ------------------
model = ANN()
model.load_state_dict(torch.load("ann_mnist1.pth", map_location="cpu"))
model.eval()

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# ------------------ UI ------------------
st.title("🧠 MNIST Digit Classifier")

st.write("Upload a handwritten digit image (0–9)")

file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# ------------------ PREDICTION ------------------
if file:
    img = Image.open(file)

    st.image(img, caption="Uploaded Image", width=150)

    img = transform(img)

    # Fix inversion (important)
    img = 1 - img

    # Normalize (same as training)
    img = (img - 0.5) / 0.5

    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs).item()

    st.subheader(f"🔢 Prediction: {pred}")

    # Confidence scores
    st.write("### Confidence Scores")
    for i, p in enumerate(probs[0]):
        st.write(f"{i}: {p.item():.4f}")

    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(range(10), probs[0].numpy())
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

# ------------------ MODEL PERFORMANCE ------------------
st.header("📊 Model Performance")

# Loss Graph
st.subheader("Loss vs Epoch")
st.image("loss.png")

# Confusion Matrix
st.subheader("Confusion Matrix")
st.image("confusion_matrix.png")

# Feature Visualization
st.subheader("Feature Visualization")
st.image("features.png")

# Classification Report
st.subheader("Classification Report")
try:
    with open("report.txt", "r") as f:
        st.text(f.read())
except:
    st.write("Report file not found")

# ------------------ INFO ------------------
st.header("ℹ️ Model Info")
st.write("Architecture: ANN (64 → 64 → 32)")
st.write("Dataset: MNIST")
st.write("Accuracy: ~96%")