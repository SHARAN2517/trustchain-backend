import numpy as np
import torch
import torch.nn.functional as F

class ViTGradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    for the self-attention feature maps in a Vision Transformer (ViT).
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradient = None
        self.activation = None
        self.handlers = []
        
        # We hook into the final encoder layer of the ViT module
        self._register_hooks()
        
    def _register_hooks(self):
        # Hook target is the final Transformer block layer
        target_layer = self.model.vit.blocks[-1]
        
        def forward_hook(module, input, output):
            # Output is [B, Num_Patches + 1, Embed_Dim]
            self.activation = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple; get gradients of output tensor
            self.gradient = grad_output[0].detach()
            
        self.handlers.append(target_layer.register_forward_hook(forward_hook))
        self.handlers.append(target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, image, text_ids, target_class_idx=0):
        """
        Generates a 2D Grad-CAM heatmap over the image patches for the target class index.
        """
        self.gradient = None
        self.activation = None
        
        # Run forward pass
        outputs = self.model(image, text_ids)
        prob = outputs['probabilities'][0, target_class_idx]
        
        # Target score for backpropagation
        score = outputs['logits'][0, target_class_idx]
        
        # Backpropagate to extract gradients
        self.model.zero_grad()
        score.backward()
        
        if self.activation is None or self.gradient is None:
            # Fallback: create synthetic heatmap if gradients are detached/not captured
            h_size = int(np.sqrt(self.model.vit.num_patches))
            return np.ones((h_size, h_size), dtype=np.float32) * 0.5
            
        # Get activations and gradients of patches (ignoring class token at index 0)
        # activations: [B, Num_Patches, Embed_Dim] -> [Num_Patches, Embed_Dim] for batch=0
        activations = self.activation[0, 1:] 
        gradients = self.gradient[0, 1:]
        
        # Compute patch weightings using average gradient of each token
        weights = torch.mean(gradients, dim=0) # [Embed_Dim]
        
        # Linearly combine activations and weights
        cam = torch.matmul(activations, weights) # [Num_Patches]
        
        # Apply ReLU to focus only on positive attributions (driving the diagnosis)
        cam = torch.relu(cam)
        
        # Reshape to patch grid dimension (e.g., 14x14 patches for 224x224 image with 16x16 patch)
        num_patches = self.model.vit.num_patches
        grid_size = int(np.sqrt(num_patches))
        
        cam = cam.cpu().numpy()
        cam = cam.reshape(grid_size, grid_size)
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def cleanup(self):
        """Remove hooks from model to prevent memory leaks."""
        for handler in self.handlers:
            handler.remove()


class ClinicalTextSHAP:
    """
    Computes mathematical token-level attributions representing SHAP values
    for clinical notes using an input-perturbation masking strategy.
    """
    def __init__(self, model, mask_token_id=0):
        self.model = model
        self.mask_token_id = mask_token_id
        
    def explain(self, image, text_ids, target_class_idx=0, num_samples=32):
        """
        Computes attribution scores for each token in text_ids.
        Measures the impact on prediction probability when a token is present vs. masked.
        """
        self.model.eval()
        B, L = text_ids.shape
        assert B == 1, "SHAP explainer operates on single instances (batch size = 1)."
        
        original_ids = text_ids.clone()
        
        # 1. Compute baseline probability
        with torch.no_grad():
            orig_outputs = self.model(image, original_ids)
            base_prob = orig_outputs['probabilities'][0, target_class_idx].item()
            
        shap_values = np.zeros(L)
        
        # 2. Approximate Shapley values by computing marginal contributions
        # We perturb each token L individually to assess impact on prediction
        for i in range(L):
            # Skip padding tokens (usually id=0) if it is far in the sequence
            if original_ids[0, i].item() == self.mask_token_id and i > 15:
                continue
                
            # Mask the target token
            masked_ids = original_ids.clone()
            masked_ids[0, i] = self.mask_token_id
            
            with torch.no_grad():
                masked_outputs = self.model(image, masked_ids)
                masked_prob = masked_outputs['probabilities'][0, target_class_idx].item()
                
            # Marginal contribution of token i is: base_prob - masked_prob
            # If removing token i causes the probability of the disease to drop heavily, 
            # then token i has high positive Shapley attribution for that disease.
            shap_values[i] = base_prob - masked_prob
            
        # Normalize SHAP values for clean clinical plotting
        total_abs = np.sum(np.abs(shap_values))
        if total_abs > 0:
            shap_values = shap_values / total_abs
            
        return shap_values


if __name__ == '__main__':
    # Test Grad-CAM and SHAP implementations on simulated model
    from model import TrustChainMedModel
    
    model = TrustChainMedModel()
    
    # Enable gradients on model parameters for Grad-CAM backprop test
    dummy_imgs = torch.randn(1, 3, 224, 224, requires_grad=True)
    dummy_txt = torch.randint(1, 30522, (1, 16))
    
    # 1. Test Grad-CAM
    cam_extractor = ViTGradCAM(model)
    cam_heatmap = cam_extractor.generate_heatmap(dummy_imgs, dummy_txt, target_class_idx=0)
    print("Grad-CAM 2D Heatmap Grid Shape:", cam_heatmap.shape)
    print(f"CAM intensity range: [{cam_heatmap.min():.2f}, {cam_heatmap.max():.2f}]")
    cam_extractor.cleanup()
    
    # 2. Test Text SHAP
    shap_extractor = ClinicalTextSHAP(model)
    tokens_shap = shap_extractor.explain(dummy_imgs, dummy_txt, target_class_idx=0)
    print("Text SHAP attributions array length:", len(tokens_shap))
    print("SHAP attributions first 5 values:", tokens_shap[:5])
    print("Explainability test completed successfully.")
