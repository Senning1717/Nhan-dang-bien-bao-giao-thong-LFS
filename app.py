import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# -------------------------
# 1. Load class names
# -------------------------
classes = pd.read_csv("Meta.csv")
class_names = classes["SignName"].tolist() if "SignName" in classes.columns else [str(i) for i in range(len(classes))]

# -------------------------
# 2. Load trained model
# -------------------------
num_classes = len(class_names)
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Load weights đã train
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location=torch.device("cpu")))
model.eval()

# -------------------------
# 3. Transform giống lúc train
# -------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------------
# 4. Streamlit UI
# -------------------------
st.title("🚦 Nhận dạng biển báo giao thông")
st.write("Upload một ảnh biển báo để dự đoán")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã upload", use_column_width=True)

    # Tiền xử lý
    img_tensor = transform(image).unsqueeze(0)  # add batch dim

    # Dự đoán
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()
        class_name = class_names[class_id]

    st.success(f"Dự đoán: {class_name} (Class ID: {class_id})")
