# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设的类别数量和图像大小
NUM_CLASSES = 43 # 从 utils_data.py 导入或硬编码
IMG_SIZE = 112      # 从 utils_data.py 导入或硬编码

class PatchEmbedding(nn.Module):
    """图像到 Patch Embedding 的转换"""
    def __init__(self, img_size=IMG_SIZE, patch_size=5, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    """轻量级 Vision Transformer 模型"""
    def __init__(self, img_size=IMG_SIZE, patch_size=5, in_channels=3, num_classes=NUM_CLASSES,
                 embed_dim=128, depth=3, num_heads=4, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True # 重要：确保输入是 (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, n_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1) # (B, n_patches + 1, embed_dim)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer_encoder(x)
        x = self.norm(x)

        # 取 CLS token 的输出用于分类
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x

def get_model(use_vit_base_config=False):
    """
    实例化并返回 ViT 模型。
    Args:
        use_vit_base_config (bool): 如果为 True，则使用接近 ViT-Base 的配置；
                                     否则使用轻量级配置。
    """
    if use_vit_base_config:
        # 接近 ViT-Base 的配置（这里调整了 patch_size 适配 IMG_SIZE 224
        # 实际 ViT-Base/16 是 patch_size=16, embed_dim=768, depth=12, num_heads=12)
        # 这里的参数是为了演示不同配置，并会比标准的ViT-Base小很多，但仍比lightweight大
        model = VisionTransformer(
            img_size=IMG_SIZE,
            patch_size=16, # 例如: 224/16 = 14 patches per dim -> 196 patches
            embed_dim=768,  # 增大嵌入维度
            depth=6,  # 增加 Transformer 层数
            num_heads=8,  # 增加注意力头数
            mlp_ratio=4.,  # 保持 MLP 层的膨胀率
            num_classes=NUM_CLASSES,
            dropout=0.1
        )
    else:
        # 轻量级配置
        model = VisionTransformer(
            img_size=IMG_SIZE,
            patch_size=5,       # 例如: 224/5 = 44 patches per dim -> 1936 patches (large!)
                                # Note: with 224 and patch_size=5, n_patches is (224//5)^2 = 44^2 = 1936.
                                # This is quite large for a "lightweight" model, consider a larger patch_size
                                # for lightweight if you want fewer patches, e.g., patch_size=16 or 32.
                                # Let's adjust for meaningful comparison with a "base" config that has fewer patches.
                                # For example, if patch_size=32, (224//32)^2 = 7^2 = 49 patches.
                                # For this example, I'll keep patch_size=5 as in your original "lightweight" example.
            embed_dim=64,       # 减小嵌入维度
            depth=2,            # 减少 Transformer 层数
            num_heads=2,        # 减少注意力头数
            mlp_ratio=2.,       # 减小 MLP 层的膨胀率
            num_classes=NUM_CLASSES,
            dropout=0.1
        )
    return model

if __name__ == '__main__':
    print(f"模型配置：输入图像为 {IMG_SIZE}x{IMG_SIZE}，类别数为 {NUM_CLASSES}。")  # 中文打印

    # 测试类似 ViT-Base 的配置
    model_base = get_model(use_vit_base_config=True)
    dummy_input_base = torch.randn(1, 3, IMG_SIZE, IMG_SIZE) # (batch_size, channels, height, width)
    output_base = model_base(dummy_input_base)
    print(f"ViT-Base 配置输出形状: {output_base.shape}") # 中文打印，输出形状应为 (batch_size, num_classes)
    
    total_params_base = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    print(f"ViT-Base 配置 - 可训练参数总数: {total_params_base:,}") # 中文打印，参数量应大于轻量级模型，反映更复杂的结构。

