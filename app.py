# ========================================================================
# STREAMLIT APP — Quantum 6-Qubit ROP Classifier with Segmentation
# ========================================================================

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import timm
from transformers import ViTModel
import pennylane as qml
import os

# ========================================================================
# 1️⃣ App Configuration
# ========================================================================
st.set_page_config(page_title="Quantum ROP Classifier", page_icon="🧠", layout="wide")
st.title("🩺 Quantum Hybrid ROP Classification (6-Qubit Model)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEGMENT_MODEL_PATH = r"C:\Users\siva\Downloads\segment.pth"
CLASSIFIER_MODEL_PATH = r"C:\Users\siva\Downloads\best_model_6qubit.pth"

# ========================================================================
# 2️⃣ Segmentation Model (U-Net)
# ========================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        conv4 = self.dconv_down4(self.maxpool(conv3))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

# ========================================================================
# 3️⃣ Quantum Circuit
# ========================================================================
def create_quantum_circuit(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        amplitude_features = inputs[:, :8]
        angle_features = inputs[:, 8:]
        qml.AmplitudeEmbedding(amplitude_features, wires=range(n_qubits), pad_with=0., normalize=True)
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(angle_features[:, i], wires=i)
            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit

# ========================================================================
# 4️⃣ Hybrid Model (ViT + CNN + Quantum)
# ========================================================================
class TwoBranchHybridTitan(nn.Module):
    def __init__(self, segmentation_model, n_qubits=6, n_layers=3):
        super().__init__()
        self.segmentation_model = segmentation_model
        for p in self.segmentation_model.parameters():
            p.requires_grad = False

        # ViT branch
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for p in self.vit.parameters():
            p.requires_grad = False
        for i in range(11, 12):
            for p in self.vit.encoder.layer[i].parameters():
                p.requires_grad = True
        for p in self.vit.embeddings.parameters():
            p.requires_grad = True

        # ResNet branch
        self.mask_cnn = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="avg")
        for p in self.mask_cnn.parameters():
            p.requires_grad = False
        orig_conv1 = self.mask_cnn.conv1
        self.mask_cnn.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.mask_cnn.conv1.weight.copy_(orig_conv1.weight.mean(dim=1, keepdim=True))
        self.mask_cnn.conv1.requires_grad = True

        # Fusion & Quantum
        fusion_dim = 768 + 512
        self.mha_refiner = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.compression = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 16),
            nn.Tanh()
        )
        self.vqc = qml.qnn.TorchLayer(create_quantum_circuit(n_qubits, n_layers), {"weights": (n_layers, n_qubits, 3)})
        self.classifier_head = nn.Linear(n_qubits, 1)

        self.register_buffer('vit_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vit_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x_vit = (x - self.vit_mean) / (self.vit_std + 1e-8)
        vit_feat = self.vit(x_vit).last_hidden_state[:, 0, :]
        with torch.no_grad():
            mask = torch.sigmoid(self.segmentation_model(x))
        mask_feat = self.mask_cnn(mask)
        fused = torch.cat([vit_feat, mask_feat], dim=1).unsqueeze(1)
        refined, _ = self.mha_refiner(fused, fused, fused)
        compressed = self.compression(refined.squeeze(1))
        q_out = self.vqc(compressed)
        return self.classifier_head(q_out), mask

# ========================================================================
# 5️⃣ Load Models
# ========================================================================
@st.cache_resource
def load_models():
    seg_model = UNet().to(DEVICE)
    seg_model.load_state_dict(torch.load(SEGMENT_MODEL_PATH, map_location=DEVICE))
    seg_model.eval()

    model = TwoBranchHybridTitan(seg_model, n_qubits=6, n_layers=3).to(DEVICE)
    state = torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

model = load_models()

# ========================================================================
# 6️⃣ Image Upload + Prediction
# ========================================================================
uploaded_img = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output, mask = model(img_tensor)
        pred = torch.sigmoid(output).item()
        pred_label = int(pred > 0.5)
        mask_np = mask.squeeze().cpu().numpy()

    class_names = ["Normal", "ROP"]
    st.markdown(f"### 🧠 Prediction: **{class_names[pred_label]}**")
    st.write(f"Confidence: `{pred:.4f}`")

    st.markdown("### 🩻 Segmentation Mask")
    st.image(mask_np, clamp=True, width=300)
