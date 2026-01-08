# similarity.py
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# -----------------------------
# Config
# -----------------------------
EMBEDDINGS_FILE = "item_embeddings.npy"
SIMILARITY_THRESHOLD = 0.85
TOP_K = 5

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load ResNet50 (feature extractor)
# -----------------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
model.to(device)

# -----------------------------
# Load embeddings
# -----------------------------
item_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
print(f"Loaded embeddings for {len(item_embeddings)} items.")

# -----------------------------
# Get embedding from image
# -----------------------------
def get_embedding(image):
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(tensor)

    return feature.squeeze().cpu().numpy()

# -----------------------------
# Cosine similarity
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# Find TOP-K similar items
# -----------------------------
def find_top_k_similar(query_image, k=TOP_K):
    query_emb = get_embedding(query_image)

    scores = []

    for item, emb in item_embeddings.items():
        sim = cosine_similarity(query_emb, emb)
        scores.append((item, float(sim)))

    # Sort by similarity (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Apply threshold + top-k
    top_results = [
        (item, score) for item, score in scores
        if score >= SIMILARITY_THRESHOLD
    ][:k]

    return top_results

# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    test_image = r"G:\My Drive\GoldSafe\Dataset\Bangle\Bangle_001\1.webp"

    results = find_top_k_similar(test_image)

    if not results:
        print("No similar items found")
    else:
        print("Top similar items:")
        for i, (item, score) in enumerate(results, start=1):
            print(f"{i}. {item} → similarity: {score:.4f}")
