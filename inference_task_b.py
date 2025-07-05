

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from glob import glob
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from B_model import ImprovedEmbeddingModel
from augmentations import AlbumentationsTransform, get_val_transforms
from config import cfg

# Set task B test directory and device
cfg.task = "B"
test_dir = "enter test directory here"  # Ensure test_dir is set correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and threshold from checkpoint
def load_model_and_threshold():
    print(f"Loading model for backbone: {cfg.backbone}")
    model = ImprovedEmbeddingModel(cfg.backbone, cfg.embedding_dim, pretrained=False).to(device)

    state = torch.load("best_B_model_complete.pth", map_location=device)
    model.load_state_dict(state['model_state_dict'], strict=False)

    threshold = state.get('optimal_threshold', 0.7)
    print(f" Loaded model. Using threshold: {threshold:.4f}")

    model.eval()
    return model, threshold

# Extract embedding from image path
def get_embedding(model, img_path, transform):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(img).squeeze(0).cpu()

# Run inference
def run_inference(test_dir):
    transform = AlbumentationsTransform(get_val_transforms())
    model, threshold = load_model_and_threshold()

    ref_embeddings = defaultdict(list)
    similarity_scores = []
    true_labels = []
    pred_labels = []

    persons = sorted(os.listdir(test_dir))
    for person in persons:
        person_dir = os.path.join(test_dir, person)
        if not os.path.isdir(person_dir): continue

        # Collect reference images
        ref_imgs = glob(os.path.join(person_dir, "*.jpg")) + glob(os.path.join(person_dir, "*.png"))
        for img_path in ref_imgs:
            ref_emb = get_embedding(model, img_path, transform)
            ref_embeddings[person].append(ref_emb)

    # Second pass: distortion queries
    for person in persons:
        distortion_dir = os.path.join(test_dir, person, "distortion")
        if not os.path.exists(distortion_dir): continue

        distorted_imgs = glob(os.path.join(distortion_dir, "*.jpg")) + glob(os.path.join(distortion_dir, "*.png"))
        for img_path in distorted_imgs:
            query_emb = get_embedding(model, img_path, transform)

            best_score = -1
            matched_person = None
            for ref_person, ref_embs in ref_embeddings.items():
                sims = [F.cosine_similarity(query_emb.unsqueeze(0), ref.unsqueeze(0), dim=1).item()
                        for ref in ref_embs]
                max_sim = max(sims)
                if max_sim > best_score:
                    best_score = max_sim
                    matched_person = ref_person

            similarity_scores.append(best_score)
            pred_labels.append(matched_person)
            true_labels.append(person)

    # Binary correctness evaluation (like training)
    is_correct = [pred == true for pred, true in zip(pred_labels, true_labels)]
    true_match = [1 if match else 0 for match in is_correct]
    pred_match = [1 if (match and score >= threshold) else 0 for match, score in zip(is_correct, similarity_scores)]

    acc = accuracy_score(true_match, pred_match)
    f1 = f1_score(true_match, pred_match)
    prec = precision_score(true_match, pred_match, zero_division=0)
    rec = recall_score(true_match, pred_match, zero_division=0)

    print("\n=== Face Verification Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("===================================\n")

if __name__ == "__main__":
    run_inference(test_dir)
