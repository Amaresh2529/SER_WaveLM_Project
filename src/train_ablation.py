import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

from dataset import UnifiedWaveLMDataset
from model import DomainAdversarialWaveLM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting ABLATION Baseline Training on {device}")
    print("WARNING: Domain Adversarial GRL is TURNED OFF for this run.")

    print("Loading full dataset...")
    full_dataset = UnifiedWaveLMDataset(root_dir="data")

    TARGET_DOMAIN = 0 
    print(f"\n--- LOCO Protocol (7 Classes) ---")
    print(f"Testing on Domain {TARGET_DOMAIN} (Unseen Data)")

    train_idx = [i for i, dom in enumerate(full_dataset.domain_labels) if dom != TARGET_DOMAIN]
    val_idx = [i for i, dom in enumerate(full_dataset.domain_labels) if dom == TARGET_DOMAIN]

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Calculate Class Weights
    train_labels = [full_dataset.emotion_labels[i] for i in train_idx]
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=7).float()
    class_counts = torch.where(class_counts == 0, torch.tensor(1.0), class_counts)
    class_weights = 1.0 / class_counts
    class_weights = (class_weights / class_weights.sum()) * 7.0 
    class_weights = class_weights.to(device)

    model = DomainAdversarialWaveLM(num_emotions=7, num_domains=4).to(device)
    
    criterion_emotion = nn.CrossEntropyLoss(weight=class_weights) 
    criterion_domain = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    num_epochs = 15
    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    # =================================================================
    # THE ABLATION: Set Domain Weight to 0.0 to disable the GRL
    # =================================================================
    LAMBDA_DOMAIN = 0.0 

    for epoch in range(num_epochs):
        model.train()
        
        running_loss = 0.0
        correct_emo = 0
        total_emo = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
        for waveforms, emo_labels, dom_labels in loop:
            waveforms = waveforms.to(device)
            emo_labels = emo_labels.to(device)
            dom_labels = dom_labels.to(device)

            optimizer.zero_grad()

            # alpha=0.0 means the GRL does nothing during the backward pass
            emo_logits, dom_logits = model(waveforms, alpha=0.0)

            loss_emo = criterion_emotion(emo_logits, emo_labels)
            loss_dom = criterion_domain(dom_logits, dom_labels)
            
            # Since LAMBDA_DOMAIN is 0.0, the model only learns emotion, not domain adaptation
            loss = loss_emo + (LAMBDA_DOMAIN * loss_dom)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(emo_logits, dim=1)
            correct_emo += (preds == emo_labels).sum().item()
            total_emo += emo_labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100.*correct_emo/total_emo)

        train_acc = 100. * correct_emo / total_emo

        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]")
            for waveforms, emo_labels, dom_labels in val_loop:
                waveforms = waveforms.to(device)
                emo_labels = emo_labels.to(device)
                
                emo_logits, _ = model(waveforms, alpha=0.0)
                
                preds = torch.argmax(emo_logits, dim=1)
                val_correct += (preds == emo_labels).sum().item()
                val_total += emo_labels.size(0)

        val_acc = 100. * val_correct / val_total
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Emotion Acc: {train_acc:.2f}% | Val Emotion Acc (Unseen Domain): {val_acc:.2f}%\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save this as a separate file so it doesn't overwrite your SOTA model!
            torch.save(model.state_dict(), "checkpoints/best_baseline_model.pth")
            print(">>> New Baseline Model Saved! <<<\n")

if __name__ == "__main__":
    main()