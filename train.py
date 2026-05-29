import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model import TrustChainMedModel
from privacy import TrustChainPrivacyEngine
from federated import SimulatedFederation
from explainability import ViTGradCAM, ClinicalTextSHAP

def run_integration_pipeline():
    print("==================================================================")
    # Highlight local system details
    print(" TRUSTCHAIN-MED AI: INTEGRATION PIPELINE RUN")
    print("==================================================================")
    
    # 1. Initialize System Parameters
    num_classes = 8
    batch_size = 4
    epochs = 3
    dataset_size = 20
    embed_dim = 768
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STAGE 1] Device selected for training: {device}")
    
    # 2. Instantiate Multimodal Network
    print("[STAGE 2] Building Vision Transformer + ClinicalBERT Multimodal Fusion Network...")
    model = TrustChainMedModel(num_classes=num_classes, embed_dim=embed_dim)
    model.to(device)
    
    # 3. Create Synthetic Dataset (Simulating ChestMNIST / Chest X-Ray and Clinical BERT Tokens)
    print(f"[STAGE 3] Generating synthetic datasets ({dataset_size} items)...")
    # Image tensors: 224x224 RGB
    synthetic_images = torch.randn(dataset_size, 3, 224, 224)
    # Text input IDs: token sequence length = 32, vocab size = 30522
    synthetic_text_ids = torch.randint(0, 30522, (dataset_size, 32))
    # Labels: 8-class multi-label conditions
    synthetic_labels = torch.randint(0, 2, (dataset_size, num_classes)).float()
    
    dataset = TensorDataset(synthetic_images, synthetic_text_ids, synthetic_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Validation split
    val_images = torch.randn(8, 3, 224, 224)
    val_text_ids = torch.randint(0, 30522, (8, 32))
    val_labels = torch.randint(0, 2, (8, num_classes)).float()
    val_dataset = TensorDataset(val_images, val_text_ids, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 4. Integrate Opacus Differential Privacy
    print("[STAGE 4] Wrapping optimizer with Differential Privacy Engine...")
    base_optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    dp_engine = TrustChainPrivacyEngine()
    # Attach differential privacy optimizer with clipping and noise
    model_priv, opt_priv, loader_priv = dp_engine.make_private(
        model=model,
        optimizer=base_optimizer,
        data_loader=train_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0
    )
    
    # 5. Core Model Training with DP-SGD
    print("[STAGE 5] Starting local secure training loop...")
    for epoch in range(1, epochs + 1):
        model_priv.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_imgs, batch_txt, batch_lbls in loader_priv:
            batch_imgs, batch_txt, batch_lbls = batch_imgs.to(device), batch_txt.to(device), batch_lbls.to(device)
            
            opt_priv.zero_grad()
            outputs = model_priv(batch_imgs, batch_txt)
            
            # Binary Cross Entropy with Logits for Multi-label classification
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs['logits'], batch_lbls)
            loss.backward()
            opt_priv.step()
            
            epoch_loss += loss.item() * batch_imgs.size(0)
            preds = (outputs['probabilities'] > 0.5).float()
            correct_predictions += (preds == batch_lbls).sum().item()
            total_predictions += batch_lbls.numel()
            
        # Calculate privacy leakage
        epsilon = dp_engine.get_privacy_spent(
            steps=epoch * len(loader_priv),
            batch_size=batch_size,
            dataset_size=dataset_size,
            noise_multiplier=1.1
        )
        
        avg_loss = epoch_loss / dataset_size
        acc = correct_predictions / total_predictions
        print(f"Epoch {epoch}/{epochs} -> Loss: {avg_loss:.4f}, Accuracy: {acc:.4f} | DP Epsilon (ε): {epsilon:.2f} at Delta (δ): {dp_engine.target_delta}")
        
    # 6. Federated Learning Simulation
    print("\n[STAGE 6] Launching decentralized Federated Learning Simulation (Flower framework)...")
    # Simulate 3 secure hospital client nodes with their own local subsets
    h1_loader = DataLoader(TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 30522, (10, 32))), batch_size=2)
    h2_loader = DataLoader(TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 30522, (10, 32))), batch_size=2)
    h3_loader = DataLoader(TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 30522, (10, 32))), batch_size=2)
    
    fl_simulation = SimulatedFederation(
        central_model=model,
        client_loaders=[h1_loader, h2_loader, h3_loader],
        val_loader=val_loader,
        lr=0.005
    )
    
    # Run 2 global model averaging rounds
    fl_simulation.run_round(1)
    fl_simulation.run_round(2)
    
    # 7. Explainability Suite Validation (Grad-CAM & SHAP)
    print("\n[STAGE 7] Instantiating visual and textual explainability engines...")
    
    # Select a single multimodal test case
    test_img = torch.randn(1, 3, 224, 224)
    # Ensure gradients are enabled for test CAM computation
    test_img.requires_grad = True
    test_txt = torch.randint(0, 30522, (1, 32))
    
    # Visual Grad-CAM on ViT attention layers
    cam_explainer = ViTGradCAM(model)
    cam_heatmap = cam_explainer.generate_heatmap(test_img, test_txt, target_class_idx=0)
    print(f"Grad-CAM Heatmap generated successfully! Dimension: {cam_heatmap.shape}")
    print(f"Highlighted focal points range from: {cam_heatmap.min():.4f} to {cam_heatmap.max():.4f}")
    cam_explainer.cleanup()
    
    # Text SHAP attributions on ClinicalBERT tokens
    shap_explainer = ClinicalTextSHAP(model)
    shap_values = shap_explainer.explain(test_img, test_txt, target_class_idx=0)
    print(f"SHAP Text attributions computed successfully! Num Tokens: {len(shap_values)}")
    print(f"Sum of attributions: {np.sum(np.abs(shap_values)):.4f}")
    
    print("\n==================================================================")
    print(" TRUSTCHAIN-MED AI PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================================")

if __name__ == '__main__':
    run_integration_pipeline()
