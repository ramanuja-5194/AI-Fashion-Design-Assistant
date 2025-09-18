# models/text_to_fashion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, CLIPTextModel, CLIPTokenizer
import clip
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from typing import List, Optional
import requests
from io import BytesIO

class TextEncoder(nn.Module):
    """Encodes text descriptions to latent vectors"""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Use CLIP text encoder
        self.clip_model, _ = clip.load("ViT-B/32")
        
        # Additional layers for fashion-specific encoding
        self.fashion_encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        
    def forward(self, text_descriptions: List[str]):
        # Tokenize and encode with CLIP
        text_tokens = clip.tokenize(text_descriptions)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Additional fashion-specific encoding
        fashion_features = self.fashion_encoder(text_features.float())
        return fashion_features

class StyleGAN2Generator(nn.Module):
    """Simplified StyleGAN2 generator for fashion images"""
    def __init__(self, z_dim=512, w_dim=512, img_size=256):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_size = img_size
        
        # Mapping network
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim),
        )
        
        # Synthesis network
        self.synthesis = nn.Sequential(
            # Start with 4x4
            nn.ConvTranspose2d(w_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, z, text_features=None):
        # Map noise to intermediate latent
        w = self.mapping(z)
        
        # Combine with text features if provided
        if text_features is not None:
            w = w + text_features
        
        # Reshape for synthesis network
        w = w.view(w.size(0), -1, 1, 1)
        
        # Generate image
        img = self.synthesis(w)
        return img

class FashionDiscriminator(nn.Module):
    """Discriminator for fashion images"""
    def __init__(self, img_size=256):
        super().__init__()
        
        self.main = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 128x128 -> 64x64
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 4x4 -> 1x1
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        return self.main(img).view(img.size(0), -1)

class TextToFashionGenerator:
    """Main class for text-to-fashion generation"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.setup_models()
        
    def setup_models(self):
        """Initialize and load pre-trained models"""
        self.text_encoder = TextEncoder().to(self.device)
        self.generator = StyleGAN2Generator().to(self.device)
        
        # Load pre-trained weights if available
        self.load_pretrained_weights()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.denormalize = transforms.Compose([
            transforms.Normalize([-1, -1, -1], [2, 2, 2]),
            transforms.ToPILImage()
        ])
    
    def load_pretrained_weights(self):
        """Load pre-trained model weights"""
        try:
            # Try to load pre-trained weights
            generator_path = 'checkpoints/text_to_fashion_generator.pth'
            if torch.cuda.is_available():
                self.generator.load_state_dict(torch.load(generator_path))
            else:
                self.generator.load_state_dict(torch.load(generator_path, map_location='cpu'))
        except:
            # Initialize with random weights if no checkpoint available
            print("No pre-trained weights found. Using randomly initialized weights.")
            pass
    
    def generate_sample_fashion(self, style="modern"):
        """Generate a sample fashion image for demo purposes"""
        # Create a simple pattern-based fashion image
        img = Image.new('RGB', (256, 256), color='white')
        import PIL.ImageDraw as ImageDraw
        
        draw = ImageDraw.Draw(img)
        
        if style == "modern":
            # Modern minimalist design
            draw.rectangle([50, 80, 206, 220], fill='black', outline='gray')
            draw.rectangle([60, 90, 196, 120], fill='white')
        elif style == "vintage":
            # Vintage floral pattern
            draw.ellipse([60, 60, 196, 196], fill='lightpink', outline='darkred')
            draw.ellipse([80, 80, 176, 176], fill='white')
        elif style == "bohemian":
            # Bohemian flowing design
            draw.polygon([(128, 60), (80, 120), (90, 180), (166, 180), (176, 120)], 
                        fill='orange', outline='brown')
        elif style == "minimalist":
            # Ultra-minimal design
            draw.line([100, 80, 156, 80], fill='black', width=3)
            draw.rectangle([120, 90, 136, 200], fill='gray')
        else:  # avant-garde
            # Abstract avant-garde
            draw.polygon([(50, 100), (100, 50), (150, 100), (200, 150), (100, 200)], 
                        fill='purple', outline='black')
        
        return img
    
    def generate(self, text_prompt: str, style: str = "modern") -> Image.Image:
        """Generate fashion design from text prompt"""
        try:
            # For demonstration, we'll generate sample images
            # In a real implementation, this would use the trained models
            generated_image = self.generate_sample_fashion(style)
            
            # Add text-based modifications
            if "red" in text_prompt.lower():
                # Modify colors based on text
                import PIL.ImageEnhance as ImageEnhance
                enhancer = ImageEnhance.Color(generated_image)
                generated_image = enhancer.enhance(1.5)
            
            return generated_image
            
        except Exception as e:
            print(f"Generation error: {e}")
            # Return a default image
            return self.generate_sample_fashion(style)
    
    def train(self, dataloader, num_epochs=100):
        """Training loop for the text-to-fashion model"""
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        discriminator = FashionDiscriminator().to(self.device)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        criterion = nn.BCELoss()
        
        for epoch in range(num_epochs):
            for i, (images, descriptions) in enumerate(dataloader):
                batch_size = images.size(0)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                # Real images
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                real_images = images.to(self.device)
                real_output = discriminator(real_images)
                d_loss_real = criterion(real_output, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size, 512).to(self.device)
                text_features = self.text_encoder(descriptions)
                fake_images = self.generator(noise, text_features)
                fake_output = discriminator(fake_images.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                fake_output = discriminator(fake_images)
                g_loss = criterion(fake_output, real_labels)
                g_loss.backward()
                optimizer_G.step()
                
                if i % 100 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                          f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
            
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save(self.generator.state_dict(), 
                          f'checkpoints/generator_epoch_{epoch}.pth')