import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import UnifiedWaveLMDataset
from model import DomainAdversarialWaveLM

# UPDATED: 7 Classes (Calm is now merged into Neutral)
EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    print("Loading dataset...")
    full_dataset = UnifiedWaveLMDataset(root_dir="data")
    TARGET_DOMAIN = 0  # RAVDESS

    val_idx = [i for i, dom in enumerate(full_dataset.domain_labels) if dom == TARGET_DOMAIN]
    val_ds = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    print(f"Validation samples (Unseen Domain): {len(val_ds)}")

    # Initialize model with 7 emotions!
    model = DomainAdversarialWaveLM(num_emotions=7, num_domains=4).to(device)
    
    try:
        model.load_state_dict(torch.load("checkpoints/best_loco_model.pth", map_location=device, weights_only=True))
        print("Successfully loaded 'best_loco_model.pth'")
    except FileNotFoundError:
        print("Error: Could not find 'checkpoints/best_loco_model.pth'.")
        return

    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for waveforms, emo_labels, _ in tqdm(val_loader, desc="Evaluating"):
            waveforms = waveforms.to(device)
            
            # Forward pass (alpha=0 since we aren't training the domain head)
            emo_logits, _ = model(waveforms, alpha=0.0)
            
            preds = torch.argmax(emo_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(emo_labels.numpy())

    # Calculate SOTA Metrics
    wa = accuracy_score(all_labels, all_preds) * 100
    uar = recall_score(all_labels, all_preds, average='macro') * 100

    print("\n" + "="*40)
    print(f"Weighted Accuracy (WA):      {wa:.2f}%")
    print(f"Unweighted Average Recall:   {uar:.2f}%")
    print("="*40 + "\n")

    # Generate and Save the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, 
                yticklabels=EMOTION_LABELS)
    
    plt.title(f"Cross-Corpus Emotion Confusion Matrix (7-Class)\n(Train: CREMA/TESS/SAVEE | Test: RAVDESS)\nUAR: {uar:.2f}%", pad=20, fontsize=14)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.tight_layout()
    
    plt.savefig("confusion_matrix_ravdess_7class.png", dpi=300)
    print("Saved beautiful confusion matrix to 'confusion_matrix_ravdess_7class.png'")

if __name__ == "__main__":
    main()