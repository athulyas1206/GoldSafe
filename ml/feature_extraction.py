import os
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

# -----------------------------
# Config
# -----------------------------
DATASET_DIR = r"G:\\My Drive\\GoldSafe\\Dataset"  # Path to dataset folder
EMBEDDINGS_FILE = "item_embeddings.npy"  # Where to save embeddings

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing for ResNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# Load pretrained ResNet50 and remove classification layer
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final layer
model.eval()
model.to(device)

# -----------------------------
# Helper function to process an image
# -----------------------------
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor_img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(tensor_img)
    feature = feature.squeeze().cpu().numpy()  # convert to numpy
    return feature

# -----------------------------
# Main: Process dataset
# -----------------------------
item_embeddings = {}  # key: category/item_id, value: embedding vector

categories = os.listdir(DATASET_DIR)
categories.sort()  # optional: for consistent ordering

for category in categories:
    category_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(category_path):
        continue

    items = os.listdir(category_path)
    items.sort()

    for item in items:
        item_path = os.path.join(category_path, item)
        if not os.path.isdir(item_path):
            continue

        # Get all images in this item folder
        images = [f for f in os.listdir(item_path) if f.endswith(".webp")]
        embeddings = []

        for img_file in images:
            img_path = os.path.join(item_path, img_file)
            try:
                emb = get_embedding(img_path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if len(embeddings) == 0:
            continue

        # Average embedding for multiple angles
        avg_embedding = np.mean(embeddings, axis=0)
        item_embeddings[f"{category}/{item}"] = avg_embedding

# -----------------------------
# Save embeddings
# -----------------------------
np.save(EMBEDDINGS_FILE, item_embeddings)
print(f"Saved embeddings for {len(item_embeddings)} items to {EMBEDDINGS_FILE}")
