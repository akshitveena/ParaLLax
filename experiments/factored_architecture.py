"""
factored_architecture.py

Implements the Factored Architecture from Appendix N of the ParaLLax paper.
The paper identifies that joint training causes a 93% content collapse due to
gradient competition in the shared bottleneck. The solution is to decouple the
content-preserving encoder (Recall@1≈0.996) from a non-competing validity readout.

This experiment defines a FactoredRiDAE model with two separate paths:
1. Reconstruction path (encoder -> bottleneck -> decoder) trained with L_reconstruct + L_MNR.
2. Validity Readout (MLP classifier) trained with cross-entropy on Type-A vs Type-B labels,
   which reads from a stop-gradient copy of the encoder features.
"""

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

# Add main directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main')))

from ridae import RiDAE, BottleneckFFN, ReasoningDecoder
from schema import Candidate
from data_pipeline import load_candidates, load_contrastive_pairs

class FactoredRiDAE(nn.Module):
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name)
        self.bottleneck = BottleneckFFN(in_dim=384, hidden_dims=[256, 128], out_dim=64)
        self.decoder = ReasoningDecoder(in_dim=64, hidden_dims=[128, 256], out_dim=384)
        
        # Validity Readout: 384 -> 128 -> 2
        self.validity_readout = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, input_texts):
        # Forward pass through encoder
        # Extract sentence embeddings
        encoder_out = self.encoder.encode(input_texts, convert_to_tensor=True)
        
        # Head 1: Reconstruction
        z = self.bottleneck(encoder_out)
        reconstruction = self.decoder(z)
        
        # Head 2: Validity Readout (Stop gradient to prevent competition)
        frozen_encoder_out = encoder_out.detach()
        validity_logits = self.validity_readout(frozen_encoder_out)
        
        return encoder_out, reconstruction, validity_logits

class JointRiDAE(nn.Module):
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name)
        self.bottleneck = BottleneckFFN(in_dim=384, hidden_dims=[256, 128], out_dim=64)
        self.decoder = ReasoningDecoder(in_dim=64, hidden_dims=[128, 256], out_dim=384)
        
        # Joint Validity Readout from bottleneck: 64 -> 32 -> 2
        self.validity_readout = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, input_texts):
        encoder_out = self.encoder.encode(input_texts, convert_to_tensor=True)
        
        # Shared bottleneck
        z = self.bottleneck(encoder_out)
        reconstruction = self.decoder(z)
        
        # Validity readout from shared bottleneck, NO stop gradient
        validity_logits = self.validity_readout(z)
        
        return encoder_out, reconstruction, validity_logits

class CandidateDataset(Dataset):
    def __init__(self, candidates):
        self.candidates = candidates
        
    def __len__(self):
        return len(self.candidates)
        
    def __getitem__(self, idx):
        return self.candidates[idx]


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    correct_recall = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            texts = [c.full_text for c in batch]
            
            # For evaluation, we compute Recall@1 over the batch.
            encoder_out, reconstruction, validity_logits = model(texts)
            
            # Recall@1: compute cosine similarity between all encoder_out and reconstruction
            similarity = F.cosine_similarity(encoder_out.unsqueeze(1), reconstruction.unsqueeze(0), dim=2)
            # similarity: [batch_size, batch_size]
            max_indices = similarity.argmax(dim=1)
            correct_recall += (max_indices == torch.arange(len(texts), device=device)).sum().item()
            total_samples += len(texts)
            
            # F1_B
            labels = []
            for c in batch:
                if c.candidate_type == 'A': labels.append(0)
                elif c.candidate_type == 'B': labels.append(1)
                else: labels.append(0) # Default for now if D is mixed, but we filter A and B below
            labels = torch.tensor(labels, device=device)
            preds = validity_logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    recall_1 = correct_recall / max(1, total_samples)
    f1_b = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return recall_1, f1_b

def collate_fn(batch):
    return batch

def train_model(model, train_loader, val_loader, device, epochs=3, is_factored=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    
    print(f"\nTraining {'FactoredRiDAE' if is_factored else 'JointRiDAE'}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            texts = [c.full_text for c in batch]
            
            # Filter batch to only A and B for validity loss (if we want pure A/B, or map D to 0)
            labels = []
            for c in batch:
                if c.candidate_type == 'A': labels.append(0)
                elif c.candidate_type == 'B': labels.append(1)
                else: labels.append(0) # D mapped to 0 or ignored. The prompt says A vs B.
            labels = torch.tensor(labels, device=device)
            
            encoder_out, reconstruction, validity_logits = model(texts)
            
            # L_reconstruct (1 - cosine)
            loss_recon = (1 - F.cosine_similarity(encoder_out, reconstruction)).mean()
            
            # L_MNR (in-batch negatives)
            similarity = F.cosine_similarity(encoder_out.unsqueeze(1), reconstruction.unsqueeze(0), dim=2) / 0.05
            target = torch.arange(len(texts), device=device)
            loss_mnr = F.cross_entropy(similarity, target)
            
            # Validity Loss
            loss_val = F.cross_entropy(validity_logits, labels)
            
            # Combined Loss
            loss = loss_recon + loss_mnr + loss_val
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        recall, f1 = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Recall@1: {recall:.4f} - F1_B: {f1:.4f}")
        
    return evaluate_model(model, val_loader, device)


def main():
    parser = argparse.ArgumentParser(description="Factored Architecture Experiment")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    try:
        candidates = load_candidates()
        # Filter A and B candidates for training
        candidates = [c for c in candidates if c.candidate_type in ('A', 'B')]
    except Exception as e:
        print(f"Could not load real data: {e}")
        candidates = []
        
    if not candidates:
        print("No A/B candidates found. Generating dummy candidates for testing.")
        candidates = [
            Candidate(candidate_type='A', full_text=f"Valid reasoning A {i}", training_label=0, id=f"A{i}") for i in range(100)
        ] + [
            Candidate(candidate_type='B', full_text=f"Right answer wrong reasoning B {i}", training_label=1, id=f"B{i}") for i in range(100)
        ]
        
    random.shuffle(candidates)
    split_idx = int(len(candidates) * 0.8)
    train_data = candidates[:split_idx]
    val_data = candidates[split_idx:]
    
    train_loader = DataLoader(CandidateDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(CandidateDataset(val_data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Train Joint Model
    joint_model = JointRiDAE()
    joint_recall, joint_f1 = train_model(joint_model, train_loader, val_loader, device, epochs=args.epochs, is_factored=False)
    
    # Train Factored Model
    factored_model = FactoredRiDAE()
    factored_recall, factored_f1 = train_model(factored_model, train_loader, val_loader, device, epochs=args.epochs, is_factored=True)
    
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"{'Metric':<20} | {'Joint Baseline':<15} | {'Factored Architecture':<20}")
    print("-" * 60)
    print(f"{'Recall@1 (Content)':<20} | {joint_recall:<15.4f} | {factored_recall:<20.4f}")
    print(f"{'F1_B (Validity)':<20} | {joint_f1:<15.4f} | {factored_f1:<20.4f}")
    print("="*50)
    print("\nConclusion: The factored architecture should maintain high Recall@1 (~0.996)")
    print("while achieving competitive F1_B, avoiding the content collapse seen in the joint model.")

if __name__ == "__main__":
    main()
