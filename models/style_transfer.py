def create_preset_style(self, preset_name):
        """Create preset style images for quick style transfer"""
        preset_styles = {
            'impressionist': self.create_impressionist_style(),
            'abstract': self.create_abstract_style(), 
            'pop_art': self.create_pop_art_style(),
            'vintage': self.create_vintage_style(),
            'watercolor': self.create_watercolor_style(),
            'geometric': self.create_geometric_style()
        }
        
        return preset_styles.get(preset_name, self.create_impressionist_style())
    
def create_impressionist_style(self):
    """Create impressionist style pattern"""
    from PIL import ImageDraw, ImageFilter
    
    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)
    
    # Create brush stroke patterns
    colors = ['#4169E1', '#FFD700', '#FF6347', '#98FB98', '#DDA0DD']
    for _ in range(100):
        x = np.random.randint(0, 256)
        y = np.random.randint(0, 256)
        w = np.random.randint(10, 30)
        h = np.random.randint(5, 15)
        color = np.random.choice(colors)
        draw.ellipse([x, y, x+w, y+h], fill=color)
    
    # Apply blur for impressionist effect
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img

def create_abstract_style(self):
    """Create abstract style pattern"""
    from PIL import ImageDraw
    
    img = Image.new('RGB', (256, 256), 'black')
    draw = ImageDraw.Draw(img)
    
    # Create abstract shapes
    colors = ['#FF0080', '#00FFFF', '#FFFF00', '#FF8000', '#8000FF']
    for _ in range(20):
        shape_type = np.random.choice(['rectangle', 'ellipse', 'polygon'])
        color = np.random.choice(colors)
        
        if shape_type == 'rectangle':
            x1, y1 = np.random.randint(0, 200, 2)
            x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape_type == 'ellipse':
            x1, y1 = np.random.randint(0, 200, 2)
            x2, y2 = x1 + np.random.randint(30, 80), y1 + np.random.randint(30, 80)
            draw.ellipse([x1, y1, x2, y2], fill=color)
    
    return img

def create_pop_art_style(self):
    """Create pop art style pattern"""
    from PIL import ImageDraw
    
    img = Image.new('RGB', (256, 256), '#FF69B4')
    draw = ImageDraw.Draw(img)
    
    # Create pop art dots pattern
    colors = ['#FFFF00', '#FF0000', '#0000FF', '#00FF00', '#FFFFFF']
    
    for x in range(0, 256, 20):
        for y in range(0, 256, 20):
            if (x + y) % 40 == 0:
                color = np.random.choice(colors)
                draw.ellipse([x, y, x+15, y+15], fill=color)
    
    return img

def create_vintage_style(self):
    """Create vintage style pattern"""
    from PIL import ImageDraw
    
    img = Image.new('RGB', (256, 256), '#F5E6D3')  # Vintage cream background
    draw = ImageDraw.Draw(img)
    
    # Create vintage floral pattern
    colors = ['#8B4513', '#D2B48C', '#CD853F', '#DEB887']
    
    # Draw vintage roses pattern
    for x in range(30, 256, 60):
        for y in range(30, 256, 60):
            color = np.random.choice(colors)
            # Draw simple rose shape
            draw.ellipse([x-10, y-10, x+10, y+10], fill=color)
            draw.ellipse([x-5, y-5, x+5, y+5], fill='#8B4513', outline=color)
    
    return img

def create_watercolor_style(self):
    """Create watercolor style pattern"""
    from PIL import ImageDraw, ImageFilter
    
    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)
    
    # Create watercolor washes
    colors = ['#E6E6FA', '#F0E68C', '#FFB6C1', '#98FB98', '#87CEEB']
    
    for _ in range(15):
        x, y = np.random.randint(0, 200, 2)
        w, h = np.random.randint(40, 120, 2)
        color = np.random.choice(colors)
        draw.ellipse([x, y, x+w, y+h], fill=color)
    
    # Apply blur for watercolor effect
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return img

def create_geometric_style(self):
    """Create geometric style pattern"""
    from PIL import ImageDraw
    
    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)
    
    # Create geometric patterns
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Draw triangular patterns
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            color = colors[(i//32 + j//32) % len(colors)]
            points = [(i, j), (i+32, j), (i+16, j+32)]
            draw.polygon(points, fill=color)
    
    return img# models/style_transfer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

class VGGFeatures(nn.Module):
    """Extract features from VGG19 for style transfer"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.ModuleList(vgg[:29]).eval()
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Define layers for content and style
        self.content_layers = [21]  # conv4_2
        self.style_layers = [0, 5, 10, 19, 28]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        
    def forward(self, x):
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.content_layers:
                features[f'content_{i}'] = x
            if i in self.style_layers:
                features[f'style_{i}'] = x
        return features

class AdaINLayer(nn.Module):
    """Adaptive Instance Normalization layer"""
    def __init__(self):
        super().__init__()
        
    def forward(self, content_features, style_features):
        # Calculate mean and std
        content_mean = torch.mean(content_features, dim=(2, 3), keepdim=True)
        content_std = torch.std(content_features, dim=(2, 3), keepdim=True)
        
        style_mean = torch.mean(style_features, dim=(2, 3), keepdim=True)
        style_std = torch.std(style_features, dim=(2, 3), keepdim=True)
        
        # Normalize content features and apply style statistics
        normalized = (content_features - content_mean) / (content_std + 1e-8)
        stylized = normalized * style_std + style_mean
        
        return stylized

class StyleTransferNetwork(nn.Module):
    """Neural style transfer network"""
    def __init__(self):
        super().__init__()
        
        # Encoder (VGG features)
        self.encoder = VGGFeatures()
        
        # AdaIN layer
        self.adain = AdaINLayer()
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample to match encoder output
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, 1, 0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, 1, 0),
            nn.Tanh()
        )
        
    def forward(self, content, style):
        # Encode both images
        content_features = self.encoder(content)
        style_features = self.encoder(style)
        
        # Apply AdaIN at the deepest layer
        content_deep = content_features['content_21']
        style_deep = style_features['style_28']
        
        stylized_features = self.adain(content_deep, style_deep)
        
        # Decode to get final image
        output = self.decoder(stylized_features)
        return output

class FashionStyleTransfer(nn.Module):
    """Fashion-specific style transfer with attention"""
    def __init__(self):
        super().__init__()
        
        # Base style transfer network
        self.style_net = StyleTransferNetwork()
        
        # Fashion attention module
        self.attention = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, content, style):
        # Generate attention map for fashion-specific regions
        attention_map = self.attention(content)
        
        # Apply style transfer
        stylized = self.style_net(content, style)
        
        # Blend based on attention
        output = content * (1 - attention_map) + stylized * attention_map
        
        return output, attention_map

class StyleTransferModel:
    """Main style transfer model class"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.setup_model()
        
    def setup_model(self):
        """Initialize the style transfer model"""
        self.model = FashionStyleTransfer().to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Image postprocessing
        self.denormalize = transforms.Compose([
            transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                               [1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage()
        ])
        
        # Load pretrained weights if available
        try:
            self.model.load_state_dict(torch.load('checkpoints/style_transfer.pth', 
                                                 map_location=self.device))
        except:
            print("No pretrained style transfer model found. Using default initialization.")
    
    def preprocess_image(self, image):
        """Preprocess image for style transfer"""
        if isinstance(image, Image.Image):
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
        return image_tensor
    
    def postprocess_image(self, tensor):
        """Convert tensor back to PIL Image"""
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0, 1)
        return self.denormalize(tensor)
    
    def transfer(self, content_image, style_image, alpha=1.0):
        """Apply style transfer between content and style images"""
        try:
            with torch.no_grad():
                # Preprocess images
                content_tensor = self.preprocess_image(content_image)
                style_tensor = self.preprocess_image(style_image)
                
                # Apply style transfer
                stylized_tensor, attention_map = self.model(content_tensor, style_tensor)
                
                # Blend with original content based on alpha
                final_tensor = alpha * stylized_tensor + (1 - alpha) * content_tensor
                
                # Convert back to PIL Image
                result_image = self.postprocess_image(final_tensor)
                
                return result_image
                
        except Exception as e:
            print(f"Style transfer error: {e}")
            # Return a simple blend as fallback
            return self.simple_style_blend(content_image, style_image)
    
    def simple_style_blend(self, content_image, style_image):
        """Simple style transfer fallback"""
        # Resize images to same size
        content_resized = content_image.resize((256, 256))
        style_resized = style_image.resize((256, 256))
        
        # Convert to arrays
        content_arr = np.array(content_resized, dtype=np.float32)
        style_arr = np.array(style_resized, dtype=np.float32)
        
        # Simple color transfer
        content_mean = np.mean(content_arr, axis=(0, 1))
        content_std = np.std(content_arr, axis=(0, 1))
        
        style_mean = np.mean(style_arr, axis=(0, 1))
        style_std = np.std(style_arr, axis=(0, 1))
        
        # Apply style statistics to content
        normalized = (content_arr - content_mean) / (content_std + 1e-8)
        stylized = normalized * style_std + style_mean
        
        # Blend 70% content, 30% style
        result = 0.7 * content_arr + 0.3 * stylized
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def train(self, dataloader, num_epochs=50):
        """Training loop for style transfer model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        mse_loss = nn.MSELoss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for i, (content_imgs, style_imgs) in enumerate(dataloader):
                content_imgs = content_imgs.to(self.device)
                style_imgs = style_imgs.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                stylized_imgs, attention_maps = self.model(content_imgs, style_imgs)
                
                # Calculate losses
                content_loss = self.calculate_content_loss(stylized_imgs, content_imgs)
                style_loss = self.calculate_style_loss(stylized_imgs, style_imgs)
                
                total_loss_batch = content_loss + style_loss
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                
                if i % 50 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                          f'Loss: {total_loss_batch.item():.4f}')
            
            print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {total_loss/len(dataloader):.4f}')
            
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), 
                          f'checkpoints/style_transfer_epoch_{epoch}.pth')
    
    def calculate_content_loss(self, stylized, content):
        """Calculate content preservation loss"""
        vgg = models.vgg19(pretrained=True).features[:21].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        vgg = vgg.to(self.device)
        
        stylized_features = vgg(stylized)
        content_features = vgg(content)
        
        return F.mse_loss(stylized_features, content_features)
    
    def calculate_style_loss(self, stylized, style):
        """Calculate style transfer loss using Gram matrices"""
        vgg = models.vgg19(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        vgg = vgg.to(self.device)
        
        def gram_matrix(features):
            batch_size, channels, height, width = features.size()
            features = features.view(batch_size * channels, height * width)
            gram = torch.mm(features, features.t())
            return gram.div(batch_size * channels * height * width)
        
        style_layers = [0, 5, 10, 19, 28]
        style_loss = 0
        
        stylized_x = stylized
        style_x = style
        
        for i, layer in enumerate(vgg):
            stylized_x = layer(stylized_x)
            style_x = layer(style_x)
            
            if i in style_layers:
                stylized_gram = gram_matrix(stylized_x)
                style_gram = gram_matrix(style_x)
                style_loss += F.mse_loss(stylized_gram, style_gram)
        
        return style_loss