# models/fashion_recommender.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import json
import random
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class UserProfileEncoder(nn.Module):
    """Encode user preferences into embeddings"""
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        # Style encodings
        self.style_embedding = nn.Embedding(20, embedding_dim // 4)
        self.color_embedding = nn.Embedding(15, embedding_dim // 4)
        self.occasion_embedding = nn.Embedding(10, embedding_dim // 4)
        
        # Preference fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, style_ids, color_ids, occasion_ids):
        # Get embeddings
        style_emb = self.style_embedding(style_ids).mean(dim=1)  # Average multiple styles
        color_emb = self.color_embedding(color_ids).mean(dim=1)
        occasion_emb = self.occasion_embedding(occasion_ids).mean(dim=1)
        
        # Concatenate embeddings
        combined = torch.cat([style_emb, color_emb, occasion_emb], dim=1)
        
        # Fuse preferences
        user_profile = self.fusion_network(combined)
        
        return user_profile

class FashionItemEncoder(nn.Module):
    """Encode fashion items into embeddings"""
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        self.item_encoder = nn.Sequential(
            nn.Linear(50, 128),  # Item features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, item_features):
        return self.item_encoder(item_features)

class CollaborativeFilteringNet(nn.Module):
    """Collaborative filtering for fashion recommendations"""
    def __init__(self, num_users=10000, num_items=50000, embedding_dim=128):
        super().__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate user and item embeddings
        combined = torch.cat([user_emb, item_emb], dim=1)
        
        # Predict rating/preference
        rating = self.fc(combined)
        
        return rating

class FashionRecommendationEngine:
    """Main recommendation engine for fashion items and outfits"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.setup_models()
        self.setup_fashion_database()
        
    def setup_models(self):
        """Initialize recommendation models"""
        self.user_encoder = UserProfileEncoder().to(self.device)
        self.item_encoder = FashionItemEncoder().to(self.device)
        self.collaborative_filter = CollaborativeFilteringNet().to(self.device)
        
        # Set to evaluation mode
        self.user_encoder.eval()
        self.item_encoder.eval()
        self.collaborative_filter.eval()
        
        # Text similarity for style matching
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Load pretrained weights if available
        self.load_pretrained_weights()
        
    def load_pretrained_weights(self):
        """Load pretrained model weights"""
        try:
            self.user_encoder.load_state_dict(
                torch.load('checkpoints/user_encoder.pth', map_location=self.device)
            )
            self.item_encoder.load_state_dict(
                torch.load('checkpoints/item_encoder.pth', map_location=self.device)
            )
            self.collaborative_filter.load_state_dict(
                torch.load('checkpoints/collaborative_filter.pth', map_location=self.device)
            )
        except:
            print("No pretrained recommendation models found. Using default initialization.")
    
    def setup_fashion_database(self):
        """Setup fashion item database and mappings"""
        # Style mappings
        self.style_to_id = {
            'casual': 0, 'formal': 1, 'bohemian': 2, 'minimalist': 3, 'vintage': 4,
            'modern': 5, 'streetwear': 6, 'romantic': 7, 'edgy': 8, 'sporty': 9,
            'preppy': 10, 'gothic': 11, 'punk': 12, 'classic': 13, 'artistic': 14,
            'elegant': 15, 'feminine': 16, 'masculine': 17, 'androgynous': 18, 'avant-garde': 19
        }
        
        # Color mappings
        self.color_to_id = {
            'black': 0, 'white': 1, 'navy': 2, 'gray': 3, 'red': 4,
            'blue': 5, 'green': 6, 'pink': 7, 'yellow': 8, 'purple': 9,
            'brown': 10, 'beige': 11, 'orange': 12, 'turquoise': 13, 'silver': 14
        }
        
        # Occasion mappings
        self.occasion_to_id = {
            'work': 0, 'casual outings': 1, 'parties': 2, 'travel': 3, 'sports': 4,
            'formal events': 5, 'date night': 6, 'shopping': 7, 'vacation': 8, 'weekend': 9
        }
        
        # Fashion item database (simplified)
        self.fashion_database = self.create_fashion_database()
        
    def create_fashion_database(self):
        """Create a comprehensive fashion item database"""
        database = {
            'tops': [
                {'name': 'Classic White Button-Down', 'style': ['formal', 'minimalist'], 'colors': ['white'], 'occasions': ['work', 'formal events']},
                {'name': 'Bohemian Flowy Blouse', 'style': ['bohemian', 'romantic'], 'colors': ['pink', 'beige'], 'occasions': ['casual outings', 'date night']},
                {'name': 'Vintage Band T-Shirt', 'style': ['vintage', 'casual'], 'colors': ['black', 'gray'], 'occasions': ['casual outings', 'weekend']},
                {'name': 'Minimalist Turtleneck', 'style': ['minimalist', 'modern'], 'colors': ['black', 'white', 'beige'], 'occasions': ['work', 'casual outings']},
                {'name': 'Streetwear Crop Top', 'style': ['streetwear', 'modern'], 'colors': ['black', 'white'], 'occasions': ['parties', 'casual outings']},
            ],
            'bottoms': [
                {'name': 'High-Waisted Black Jeans', 'style': ['casual', 'modern'], 'colors': ['black'], 'occasions': ['casual outings', 'weekend']},
                {'name': 'Pleated Midi Skirt', 'style': ['romantic', 'feminine'], 'colors': ['navy', 'beige'], 'occasions': ['work', 'date night']},
                {'name': 'Wide-Leg Trousers', 'style': ['formal', 'elegant'], 'colors': ['navy', 'black', 'gray'], 'occasions': ['work', 'formal events']},
                {'name': 'Distressed Denim Shorts', 'style': ['casual', 'streetwear'], 'colors': ['blue'], 'occasions': ['casual outings', 'weekend']},
                {'name': 'Vintage High-Waisted Shorts', 'style': ['vintage', 'romantic'], 'colors': ['white', 'beige'], 'occasions': ['casual outings', 'vacation']},
            ],
            'dresses': [
                {'name': 'Little Black Dress', 'style': ['classic', 'elegant'], 'colors': ['black'], 'occasions': ['parties', 'date night', 'formal events']},
                {'name': 'Maxi Bohemian Dress', 'style': ['bohemian', 'romantic'], 'colors': ['green', 'brown', 'beige'], 'occasions': ['casual outings', 'vacation']},
                {'name': 'Minimalist Shift Dress', 'style': ['minimalist', 'modern'], 'colors': ['white', 'beige', 'gray'], 'occasions': ['work', 'casual outings']},
                {'name': 'Vintage Floral Sundress', 'style': ['vintage', 'feminine'], 'colors': ['pink', 'yellow'], 'occasions': ['casual outings', 'date night']},
                {'name': 'Bodycon Party Dress', 'style': ['modern', 'edgy'], 'colors': ['red', 'black'], 'occasions': ['parties', 'date night']},
            ],
            'outerwear': [
                {'name': 'Classic Trench Coat', 'style': ['classic', 'elegant'], 'colors': ['beige', 'navy'], 'occasions': ['work', 'formal events']},
                {'name': 'Oversized Denim Jacket', 'style': ['casual', 'streetwear'], 'colors': ['blue'], 'occasions': ['casual outings', 'weekend']},
                {'name': 'Leather Biker Jacket', 'style': ['edgy', 'modern'], 'colors': ['black'], 'occasions': ['parties', 'date night']},
                {'name': 'Cozy Cardigan', 'style': ['casual', 'romantic'], 'colors': ['beige', 'pink', 'gray'], 'occasions': ['casual outings', 'work']},
            ]
        }
        return database
    
    def encode_user_preferences(self, preferences: Dict) -> torch.Tensor:
        """Convert user preferences to tensor encoding"""
        # Convert style preferences to IDs
        style_ids = [self.style_to_id.get(style.lower(), 0) for style in preferences.get('styles', ['casual'])]
        color_ids = [self.color_to_id.get(color.lower(), 0) for color in preferences.get('colors', ['black'])]
        occasion_ids = [self.occasion_to_id.get(occ.lower(), 0) for occ in preferences.get('occasions', ['casual outings'])]
        
        # Pad or truncate to fixed length
        style_ids = (style_ids + [0] * 5)[:5]
        color_ids = (color_ids + [0] * 5)[:5]
        occasion_ids = (occasion_ids + [0] * 3)[:3]
        
        # Convert to tensors
        style_tensor = torch.tensor([style_ids], dtype=torch.long).to(self.device)
        color_tensor = torch.tensor([color_ids], dtype=torch.long).to(self.device)
        occasion_tensor = torch.tensor([occasion_ids], dtype=torch.long).to(self.device)
        
        return style_tensor, color_tensor, occasion_tensor
    
    def calculate_item_similarity(self, user_preferences: Dict, item: Dict) -> float:
        """Calculate similarity between user preferences and fashion item"""
        score = 0.0
        total_weight = 0.0
        
        # Style matching
        user_styles = set(style.lower() for style in user_preferences.get('styles', []))
        item_styles = set(style.lower() for style in item.get('style', []))
        style_overlap = len(user_styles.intersection(item_styles))
        if user_styles:
            score += (style_overlap / len(user_styles)) * 0.4
            total_weight += 0.4
        
        # Color matching
        user_colors = set(color.lower() for color in user_preferences.get('colors', []))
        item_colors = set(color.lower() for color in item.get('colors', []))
        color_overlap = len(user_colors.intersection(item_colors))
        if user_colors:
            score += (color_overlap / len(user_colors)) * 0.3
            total_weight += 0.3
        
        # Occasion matching
        user_occasions = set(occ.lower() for occ in user_preferences.get('occasions', []))
        item_occasions = set(occ.lower() for occ in item.get('occasions', []))
        occasion_overlap = len(user_occasions.intersection(item_occasions))
        if user_occasions:
            score += (occasion_overlap / len(user_occasions)) * 0.3
            total_weight += 0.3
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def generate_outfit_combinations(self, recommended_items: List[Dict]) -> List[Dict]:
        """Generate complete outfit combinations from recommended items"""
        outfits = []
        
        # Group items by category
        tops = [item for item in recommended_items if item['category'] == 'tops']
        bottoms = [item for item in recommended_items if item['category'] == 'bottoms']
        dresses = [item for item in recommended_items if item['category'] == 'dresses']
        outerwear = [item for item in recommended_items if item['category'] == 'outerwear']
        
        # Generate top + bottom combinations
        for top in tops[:3]:  # Limit combinations
            for bottom in bottoms[:3]:
                outfit = {
                    'name': f"{top['name']} + {bottom['name']}",
                    'description': f"Pair the {top['name'].lower()} with {bottom['name'].lower()} for a stylish look",
                    'items': [top['name'], bottom['name']],
                    'occasion': self.get_common_occasions(top['occasions'], bottom['occasions'])[0],
                    'style_score': (top['similarity'] + bottom['similarity']) / 2
                }
                
                # Add outerwear if suitable
                suitable_outerwear = [o for o in outerwear if self.styles_compatible(top['style'] + bottom['style'], o['style'])]
                if suitable_outerwear:
                    best_outerwear = max(suitable_outerwear, key=lambda x: x['similarity'])
                    outfit['items'].append(best_outerwear['name'])
                    outfit['description'] += f" and layer with the {best_outerwear['name'].lower()}"
                
                outfits.append(outfit)
        
        # Add dress-based outfits
        for dress in dresses[:2]:
            outfit = {
                'name': f"{dress['name']} Outfit",
                'description': f"Wear the {dress['name'].lower()} for an effortless, put-together look",
                'items': [dress['name']],
                'occasion': dress['occasions'][0] if dress['occasions'] else 'casual outings',
                'style_score': dress['similarity']
            }
            
            # Add suitable outerwear
            suitable_outerwear = [o for o in outerwear if self.styles_compatible(dress['style'], o['style'])]
            if suitable_outerwear:
                best_outerwear = max(suitable_outerwear, key=lambda x: x['similarity'])
                outfit['items'].append(best_outerwear['name'])
                outfit['description'] += f" with the {best_outerwear['name'].lower()}"
            
            outfits.append(outfit)
        
        # Sort by style score and return top outfits
        outfits.sort(key=lambda x: x['style_score'], reverse=True)
        return outfits[:6]
    
    def get_common_occasions(self, occasions1: List[str], occasions2: List[str]) -> List[str]:
        """Get common occasions between two item lists"""
        common = list(set(occasions1).intersection(set(occasions2)))
        return common if common else occasions1
    
    def styles_compatible(self, styles1: List[str], styles2: List[str]) -> bool:
        """Check if two style lists are compatible"""
        # Define compatible style groups
        compatible_groups = [
            ['casual', 'streetwear', 'modern'],
            ['formal', 'classic', 'elegant', 'minimalist'],
            ['bohemian', 'romantic', 'vintage'],
            ['edgy', 'modern', 'avant-garde'],
        ]
        
        for group in compatible_groups:
            if any(style in group for style in styles1) and any(style in group for style in styles2):
                return True
        
        return len(set(styles1).intersection(set(styles2))) > 0
    
    def generate_style_tips(self, user_preferences: Dict, recommended_items: List[Dict]) -> List[str]:
        """Generate personalized style tips"""
        tips = []
        
        user_styles = user_preferences.get('styles', [])
        user_colors = user_preferences.get('colors', [])
        body_type = user_preferences.get('body_type', '')
        
        # Style-based tips
        if 'minimalist' in user_styles:
            tips.append("Stick to a neutral color palette and invest in quality basics that can be mixed and matched.")
        if 'bohemian' in user_styles:
            tips.append("Layer different textures and don't be afraid to mix patterns for that free-spirited boho look.")
        if 'formal' in user_styles:
            tips.append("Focus on well-tailored pieces and classic silhouettes that never go out of style.")
        
        # Color-based tips
        if 'black' in user_colors and 'white' in user_colors:
            tips.append("Your monochrome palette is timeless! Add one pop of color as an accent for visual interest.")
        if len(set(user_colors)) > 5:
            tips.append("You love variety in colors! Try the 60-30-10 rule: 60% neutral, 30% secondary color, 10% accent color.")
        
        # Body type tips
        if body_type and body_type != 'Not specified':
            if body_type.lower() == 'petite':
                tips.append("Vertical lines and high-waisted bottoms will elongate your silhouette beautifully.")
            elif body_type.lower() == 'tall':
                tips.append("You can pull off bold patterns and horizontal lines that others might avoid.")
            elif body_type.lower() == 'hourglass':
                tips.append("Emphasize your waist with belted styles and fitted silhouettes.")
        
        # General styling tips
        tips.extend([
            "Invest in a few quality pieces rather than many cheap items for a more polished wardrobe.",
            "Don't forget accessories! They can completely transform a basic outfit.",
            "When in doubt, fit is everything - even expensive clothes look cheap if they don't fit well."
        ])
        
        return tips[:4]  # Return top 4 tips
    
    def get_trend_alerts(self, user_preferences: Dict) -> List[str]:
        """Generate personalized trend alerts"""
        alerts = []
        
        user_styles = [style.lower() for style in user_preferences.get('styles', [])]
        
        # Style-specific trend alerts
        if 'minimalist' in user_styles:
            alerts.append("Oversized blazers are trending for minimalist wardrobes this season!")
        if 'bohemian' in user_styles:
            alerts.append("Crochet and macrame details are making a comeback in boho fashion!")
        if 'streetwear' in user_styles:
            alerts.append("Cargo pants and utility wear are dominating streetwear trends!")
        if 'vintage' in user_styles:
            alerts.append("90s slip dresses and Y2K-inspired pieces are having a major moment!")
        
        # General trend alerts
        alerts.extend([
            "Sustainable fashion is more important than ever - look for eco-friendly brands!",
            "Statement sleeves are everywhere - from puff sleeves to bell sleeves!",
            "Color-blocking is back! Don't be afraid to pair bold, contrasting colors."
        ])
        
        return alerts[:4]
    
    def recommend(self, user_preferences: Dict, style_history: List[str] = None) -> Dict:
        """Generate comprehensive fashion recommendations"""
        try:
            # Get recommended items for each category
            all_recommendations = []
            
            for category, items in self.fashion_database.items():
                for item in items:
                    similarity = self.calculate_item_similarity(user_preferences, item)
                    if similarity > 0.3:  # Threshold for relevance
                        all_recommendations.append({
                            **item,
                            'category': category,
                            'similarity': similarity
                        })
            
            # Sort by similarity
            all_recommendations.sort(key=lambda x: x['similarity'], reverse=True)
            top_recommendations = all_recommendations[:15]
            
            # Generate outfit combinations
            outfits = self.generate_outfit_combinations(top_recommendations)
            
            # Create shopping list (top individual items)
            shopping_list = []
            for item in top_recommendations[:8]:
                reason = f"Matches your {', '.join(item['style'])} style preference"
                shopping_list.append({
                    'item': item['name'],
                    'reason': reason,
                    'category': item['category']
                })
            
            # Generate style tips and trend alerts
            style_tips = self.generate_style_tips(user_preferences, top_recommendations)
            trend_alerts = self.get_trend_alerts(user_preferences)
            
            return {
                'outfits': outfits,
                'shopping_list': shopping_list,
                'style_tips': style_tips,
                'trend_alerts': trend_alerts,
                'recommended_items': top_recommendations[:10]
            }
            
        except Exception as e:
            print(f"Recommendation error: {e}")
            return self.get_default_recommendations()
    
    def get_default_recommendations(self) -> Dict:
        """Return default recommendations when analysis fails"""
        return {
            'outfits': [
                {
                    'name': 'Classic Casual',
                    'description': 'A timeless combination perfect for everyday wear',
                    'occasion': 'casual outings',
                    'items': ['White Button-Down', 'Dark Jeans', 'Sneakers']
                },
                {
                    'name': 'Office Ready',
                    'description': 'Professional yet comfortable for work',
                    'occasion': 'work',
                    'items': ['Blazer', 'Trousers', 'Blouse']
                }
            ],
            'shopping_list': [
                {'item': 'Quality White T-Shirt', 'reason': 'Versatile wardrobe staple'},
                {'item': 'Well-Fitting Jeans', 'reason': 'Essential casual bottom'},
                {'item': 'Little Black Dress', 'reason': 'Perfect for multiple occasions'}
            ],
            'style_tips': [
                'Invest in quality basics that can be mixed and matched',
                'Fit is more important than following every trend',
                'Accessories can transform any basic outfit'
            ],
            'trend_alerts': [
                'Sustainable fashion is becoming increasingly important',
                'Neutral tones continue to dominate this season'
            ]
        }
    
    def train(self, dataloader, num_epochs=50):
        """Training loop for recommendation models"""
        optimizer_user = torch.optim.Adam(self.user_encoder.parameters(), lr=0.001)
        optimizer_item = torch.optim.Adam(self.item_encoder.parameters(), lr=0.001)
        optimizer_cf = torch.optim.Adam(self.collaborative_filter.parameters(), lr=0.001)
        
        mse_loss = nn.MSELoss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for i, batch in enumerate(dataloader):
                # Unpack batch (user_preferences, item_features, ratings)
                user_prefs, item_features, ratings = batch
                
                user_prefs = user_prefs.to(self.device)
                item_features = item_features.to(self.device)
                ratings = ratings.to(self.device)
                
                # Forward pass
                optimizer_user.zero_grad()
                optimizer_item.zero_grad()
                optimizer_cf.zero_grad()
                
                # Encode user preferences and items
                user_embeddings = self.user_encoder(user_prefs)
                item_embeddings = self.item_encoder(item_features)
                
                # Predict ratings
                predicted_ratings = self.collaborative_filter(user_embeddings, item_embeddings)
                
                # Calculate loss
                loss = mse_loss(predicted_ratings.squeeze(), ratings)
                loss.backward()
                
                optimizer_user.step()
                optimizer_item.step()
                optimizer_cf.step()
                
                total_loss += loss.item()
                
                if i % 50 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                          f'Loss: {loss.item():.4f}')
            
            print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {total_loss/len(dataloader):.4f}')
            
            # Save checkpoints
            if epoch % 10 == 0:
                torch.save(self.user_encoder.state_dict(), 
                          f'checkpoints/user_encoder_epoch_{epoch}.pth')
                torch.save(self.item_encoder.state_dict(), 
                          f'checkpoints/item_encoder_epoch_{epoch}.pth')
                torch.save(self.collaborative_filter.state_dict(), 
                          f'checkpoints/collaborative_filter_epoch_{epoch}.pth')