import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformerExtractor(nn.Module):
    """
    Feature extractor representing a Vision Transformer (ViT-B/16).
    In production, this wraps transformers.ViTModel.
    Here we implement a robust PyTorch representation that simulates the patches and self-attention blocks.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=4, num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch projection
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position & class token embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B = x.shape[0]
        # Patch projection
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        x = x + self.pos_embed
        
        # Encoder passes
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        # Return CLS token embedding (overall representation) and patch tokens (for Grad-CAM)
        return x[:, 0], x[:, 1:]

class ClinicalBERTExtractor(nn.Module):
    """
    Feature extractor representing ClinicalBERT.
    In production, this wraps transformers.AutoModel for ClinicalBERT.
    Here we implement a robust transformer encoder layer that models word embeddings and contextual dependencies.
    """
    def __init__(self, vocab_size=30522, embed_dim=768, depth=4, num_heads=8):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(512, embed_dim)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, input_ids):
        B, L = input_ids.shape
        positions = torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        x = self.word_embeddings(input_ids) + self.pos_embeddings(positions)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        # Return pooled CLS token (first token) and contextual sequence tokens (for SHAP / explanation)
        return x[:, 0], x

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Layer that aligns visual features (ViT patches) 
    with textual clinical features (ClinicalBERT sequence).
    """
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_txt = nn.LayerNorm(embed_dim)
        
    def forward(self, img_cls, txt_seq):
        # Multi-head attention where Image CLS is query, Text Sequence is key and value
        # img_cls: [B, embed_dim] -> [B, 1, embed_dim]
        q = img_cls.unsqueeze(1)
        k = txt_seq
        v = txt_seq
        
        attn_out, attn_weights = self.multihead_attn(q, k, v)
        # Squeeze back query dimension and add residual
        fused = self.norm_img(img_cls + attn_out.squeeze(1))
        return fused, attn_weights

class TrustChainMedModel(nn.Module):
    """
    Complete Multimodal TrustChain-Med AI Architecture.
    Fuses ViT image features and ClinicalBERT text features.
    Outputs multi-label disease probabilities.
    """
    def __init__(self, num_classes=8, embed_dim=768):
        super().__init__()
        self.vit = VisionTransformerExtractor(embed_dim=embed_dim)
        self.clinical_bert = ClinicalBERTExtractor(embed_dim=embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim)
        
        # Classification Head (Multi-label diagnosis)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, images, clinical_text_ids):
        # Extract features
        img_cls, img_patches = self.vit(images)
        txt_cls, txt_seq = self.clinical_bert(clinical_text_ids)
        
        # Perform cross-attention fusion of image and text sequence
        fused_img_context, attn_weights = self.fusion(img_cls, txt_seq)
        
        # Concatenate fused visual context and text CLS embedding
        multimodal_rep = torch.cat((fused_img_context, txt_cls), dim=-1)
        
        # Predict disease classes
        logits = self.classifier(multimodal_rep)
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'image_patches': img_patches,
            'text_sequence': txt_seq,
            'attention_weights': attn_weights
        }

if __name__ == '__main__':
    # Dry run test
    model = TrustChainMedModel()
    dummy_imgs = torch.randn(2, 3, 224, 224)
    dummy_txt = torch.randint(0, 30522, (2, 64))
    
    outputs = model(dummy_imgs, dummy_txt)
    print("Multi-label Probabilities Shape:", outputs['probabilities'].shape)
    print("Cross-Attention Weights Shape:", outputs['attention_weights'].shape)
    print("Dry run completed successfully.")
