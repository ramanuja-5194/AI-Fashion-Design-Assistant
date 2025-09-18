import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import clip
import numpy as np
import requests
from io import BytesIO
import base64
import json
import os
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.text_to_fashion import TextToFashionGenerator
from models.style_transfer import StyleTransferModel
from models.trend_predictor import TrendPredictor
from models.fashion_recommender import FashionRecommendationEngine
from utils.image_utils import preprocess_image, postprocess_image
from utils.data_loader import FashionDataLoader

class FashionDesignAssistant:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        
    def setup_models(self):
        """Initialize all models"""
        st.write("Loading AI models... Please wait.")
        
        # Load CLIP model for text-image understanding
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Initialize custom models
        self.text_to_fashion = TextToFashionGenerator(self.device)
        self.style_transfer = StyleTransferModel(self.device)
        self.trend_predictor = TrendPredictor(self.device)
        self.recommender = FashionRecommendationEngine(self.device)
        
        st.success("All models loaded successfully!")
    
    def generate_fashion_from_text(self, text_prompt: str, style: str = "modern") -> Image.Image:
        """Generate fashion design from text description"""
        return self.text_to_fashion.generate(text_prompt, style)
    
    def apply_style_transfer(self, content_img: Image.Image, style_img: Image.Image) -> Image.Image:
        """Apply artistic style transfer to fashion items"""
        return self.style_transfer.transfer(content_img, style_img)
    
    def predict_trends(self, images: List[Image.Image]) -> dict:
        """Predict fashion trends from images"""
        return self.trend_predictor.predict(images)
    
    def get_fashion_recommendations(self, user_preferences: dict, style_history: List[str]) -> List[str]:
        """Get personalized fashion recommendations"""
        return self.recommender.recommend(user_preferences, style_history)

def main():
    st.set_page_config(
        page_title="AI Fashion Design Assistant",
        page_icon="👗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎨 AI-Powered Fashion Design Assistant")
    st.markdown("*Transform your fashion ideas into reality using advanced AI*")
    
    # Initialize the assistant
    if 'fashion_assistant' not in st.session_state:
        st.session_state.fashion_assistant = FashionDesignAssistant()
    
    assistant = st.session_state.fashion_assistant
    
    # Sidebar for feature selection
    st.sidebar.title("Features")
    feature = st.sidebar.selectbox(
        "Choose a feature:",
        ["Text-to-Fashion Generation", "Artistic Style Transfer", "Trend Prediction", "Fashion Recommendations"]
    )
    
    if feature == "Text-to-Fashion Generation":
        st.header("📝 Text-to-Fashion Generation")
        st.markdown("*Generate unique fashion designs from your text descriptions*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_prompt = st.text_area(
                "Describe your fashion design:",
                placeholder="A flowing maxi dress with floral patterns and bell sleeves",
                help="Be descriptive! Include colors, patterns, style elements, and occasion."
            )
            
            style_option = st.selectbox(
                "Choose design style:",
                ["modern", "vintage", "bohemian", "minimalist", "avant-garde", "romantic", "edgy"]
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                color_emphasis = st.slider("Color Intensity", 0.5, 2.0, 1.0, 0.1)
            with col_b:
                creativity_level = st.slider("Creativity Level", 0.1, 1.0, 0.7, 0.1)
            
            if st.button("Generate Design", type="primary"):
                if text_prompt:
                    with st.spinner("Generating fashion design..."):
                        generated_image = assistant.generate_fashion_from_text(text_prompt, style_option)
                        st.session_state.generated_fashion = generated_image
        
        with col2:
            if 'generated_fashion' in st.session_state:
                st.image(st.session_state.generated_fashion, caption="Generated Fashion Design")
                
                col_x, col_y = st.columns(2)
                with col_x:
                    # Download button
                    img_buffer = BytesIO()
                    st.session_state.generated_fashion.save(img_buffer, format='PNG')
                    st.download_button(
                        "Download Design",
                        img_buffer.getvalue(),
                        "fashion_design.png",
                        "image/png"
                    )
                
                with col_y:
                    # Generate variations
                    if st.button("Generate Variation"):
                        with st.spinner("Creating variation..."):
                            variation = assistant.generate_fashion_from_text(text_prompt, style_option)
                            st.session_state.generated_fashion = variation
                            st.rerun()
    
    elif feature == "Artistic Style Transfer":
        st.header("🎨 Artistic Style Transfer")
        st.markdown("*Apply artistic styles to your fashion designs*")
        
        # Style presets
        st.subheader("Quick Style Presets")
        preset_cols = st.columns(4)
        
        with preset_cols[0]:
            if st.button("🌸 Impressionist"):
                st.session_state.selected_preset = "impressionist"
        with preset_cols[1]:
            if st.button("🌊 Abstract"):
                st.session_state.selected_preset = "abstract"
        with preset_cols[2]:
            if st.button("✨ Pop Art"):
                st.session_state.selected_preset = "pop_art"
        with preset_cols[3]:
            if st.button("🎭 Vintage"):
                st.session_state.selected_preset = "vintage"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Fashion Item")
            content_file = st.file_uploader(
                "Upload fashion item:", 
                type=['png', 'jpg', 'jpeg'], 
                key="content",
                help="Upload a clothing item or fashion design"
            )
            if content_file:
                content_img = Image.open(content_file)
                st.image(content_img, caption="Original Fashion Item")
        
        with col2:
            st.subheader("Artistic Style")
            style_option = st.radio(
                "Choose style source:",
                ["Upload Style Image", "Use Preset Style"]
            )
            
            if style_option == "Upload Style Image":
                style_file = st.file_uploader(
                    "Upload artistic style:", 
                    type=['png', 'jpg', 'jpeg'], 
                    key="style",
                    help="Upload an artwork or pattern for style reference"
                )
                if style_file:
                    style_img = Image.open(style_file)
                    st.image(style_img, caption="Style Reference")
            else:
                if 'selected_preset' in st.session_state:
                    st.success(f"Using {st.session_state.selected_preset} preset style")
                    # Create preset style image
                    style_img = assistant.style_transfer.create_preset_style(st.session_state.selected_preset)
                    st.image(style_img, caption=f"{st.session_state.selected_preset.title()} Style")
        
        with col3:
            st.subheader("Styled Result")
            
            # Style transfer controls
            intensity = st.slider("Style Intensity", 0.1, 1.0, 0.7, 0.1)
            preserve_colors = st.checkbox("Preserve Original Colors", value=False)
            
            if st.button("Apply Style Transfer", type="primary"):
                if content_file and (style_file if style_option == "Upload Style Image" else 'selected_preset' in st.session_state):
                    with st.spinner("Applying artistic style..."):
                        if style_option == "Upload Style Image":
                            style_img = Image.open(style_file)
                        else:
                            style_img = assistant.style_transfer.create_preset_style(st.session_state.selected_preset)
                        
                        result = assistant.apply_style_transfer(content_img, style_img)
                        st.image(result, caption="Stylized Fashion Item")
                        st.session_state.style_result = result
                        
                        # Download button for result
                        img_buffer = BytesIO()
                        result.save(img_buffer, format='PNG')
                        st.download_button(
                            "Download Styled Design",
                            img_buffer.getvalue(),
                            "styled_fashion.png",
                            "image/png"
                        )
    
    elif feature == "Trend Prediction":
        st.header("📈 Fashion Trend Prediction")
        st.markdown("*Analyze fashion trends from images and market data*")
        
        # Trend analysis options
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Image-based Trend Analysis", "Market Trend Simulation", "Seasonal Predictions"]
        )
        
        if analysis_type == "Image-based Trend Analysis":
            uploaded_files = st.file_uploader(
                "Upload fashion images for trend analysis:",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload 3-20 fashion images for comprehensive analysis"
            )
            
            if uploaded_files:
                images = [Image.open(file) for file in uploaded_files]
                
                # Display uploaded images
                st.subheader(f"Analyzing {len(images)} Fashion Images")
                cols = st.columns(min(len(images), 6))
                for i, img in enumerate(images[:6]):
                    with cols[i]:
                        st.image(img, caption=f"Image {i+1}")
                
                if len(images) > 6:
                    st.info(f"+ {len(images) - 6} more images")
                
                if st.button("Analyze Fashion Trends", type="primary"):
                    with st.spinner("Analyzing fashion trends..."):
                        trends = assistant.predict_trends(images)
                        
                        st.subheader("🔍 Trend Analysis Results")
                        
                        # Create metrics display
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Trend Score", f"{trends.get('trend_score', 0):.2f}", "High Potential")
                        with metric_cols[1]:
                            st.metric("Style Diversity", f"{len(trends.get('styles', []))}", "Categories")
                        with metric_cols[2]:
                            st.metric("Color Variety", f"{len(trends.get('colors', []))}", "Dominant Colors")
                        with metric_cols[3]:
                            st.metric("Season Match", trends.get('season', 'Unknown'), "Predicted")
                        
                        # Detailed analysis
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🎨 Dominant Colors:**")
                            for i, color in enumerate(trends.get('colors', [])[:5]):
                                st.markdown(f"**{i+1}.** {color.title()}")
                            
                            st.markdown("**👕 Style Categories:**")
                            for i, style in enumerate(trends.get('styles', [])[:5]):
                                st.markdown(f"**{i+1}.** {style.title()}")
                        
                        with col2:
                            st.markdown("**🔳 Trending Patterns:**")
                            for i, pattern in enumerate(trends.get('patterns', [])[:5]):
                                st.markdown(f"**{i+1}.** {pattern.title()}")
                            
                            st.markdown(f"**🗓️ Seasonal Prediction:** **{trends.get('season', 'Unknown').title()}**")
                            
                            # Trend insights
                            trend_score = trends.get('trend_score', 0.5)
                            if trend_score > 0.7:
                                st.success("🔥 High trend potential - Great for new collections!")
                            elif trend_score > 0.4:
                                st.info("📈 Moderate trend potential - Consider for niche markets")
                            else:
                                st.warning("📉 Low trend potential - May be too niche")
        
        elif analysis_type == "Market Trend Simulation":
            st.subheader("Market Trend Simulation")
            st.info("Simulate market trends based on historical data and current indicators")
            
            # Simulation parameters
            col1, col2 = st.columns(2)
            with col1:
                target_market = st.selectbox(
                    "Target Market:",
                    ["Young Adults (18-25)", "Professionals (25-35)", "Mature (35+)", "Teenagers (13-18)"]
                )
                season = st.selectbox("Season:", ["Spring", "Summer", "Fall", "Winter"])
            
            with col2:
                price_range = st.selectbox(
                    "Price Range:",
                    ["Budget ($10-50)", "Mid-range ($50-200)", "Premium ($200-500)", "Luxury ($500+)"]
                )
                style_focus = st.selectbox(
                    "Style Focus:",
                    ["Casual", "Formal", "Streetwear", "Athletic", "Boho", "Minimalist"]
                )
            
            if st.button("Generate Market Trend Report"):
                with st.spinner("Analyzing market data..."):
                    # Simulate market analysis
                    market_trends = assistant.trend_predictor.simulate_market_trends(
                        target_market, season, price_range, style_focus
                    )
                    
                    st.success("Market Analysis Complete!")
                    
                    # Display market insights
                    st.subheader("📊 Market Trend Insights")
                    
                    for trend, confidence in market_trends.items():
                        if confidence > 0.7:
                            st.success(f"✅ **{trend}**: High confidence ({confidence:.0%})")
                        elif confidence > 0.5:
                            st.info(f"📈 **{trend}**: Moderate confidence ({confidence:.0%})")
                        else:
                            st.warning(f"⚠️ **{trend}**: Low confidence ({confidence:.0%})")
        
        else:  # Seasonal Predictions
            st.subheader("Seasonal Fashion Predictions")
            
            # Get seasonal predictions
            seasons = ["Spring 2024", "Summer 2024", "Fall 2024", "Winter 2024"]
            selected_season = st.selectbox("Select Season for Prediction:", seasons)
            
            if st.button("Get Seasonal Predictions"):
                with st.spinner("Generating seasonal predictions..."):
                    seasonal_data = assistant.trend_predictor.get_seasonal_predictions(selected_season)
                    
                    st.subheader(f"🗓️ {selected_season} Fashion Predictions")
                    
                    tabs = st.tabs(["Colors", "Styles", "Patterns", "Materials"])
                    
                    with tabs[0]:
                        st.markdown("**Trending Colors:**")
                        for color in seasonal_data['colors']:
                            st.markdown(f"• {color}")
                    
                    with tabs[1]:
                        st.markdown("**Popular Styles:**")
                        for style in seasonal_data['styles']:
                            st.markdown(f"• {style}")
                    
                    with tabs[2]:
                        st.markdown("**Trending Patterns:**")
                        for pattern in seasonal_data['patterns']:
                            st.markdown(f"• {pattern}")
                    
                    with tabs[3]:
                        st.markdown("**Key Materials:**")
                        for material in seasonal_data['materials']:
                            st.markdown(f"• {material}")
    
    elif feature == "Fashion Recommendations":
        st.header("💡 Personalized Fashion Recommendations")
        st.markdown("*Get AI-powered fashion suggestions based on your style*")
        
        # User preferences
        st.subheader("Tell us about your style preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            preferred_styles = st.multiselect(
                "Preferred Styles:",
                ["Casual", "Formal", "Bohemian", "Minimalist", "Vintage", "Modern", "Streetwear", "Romantic"],
                default=["Casual", "Modern"]
            )
            
            preferred_colors = st.multiselect(
                "Favorite Colors:",
                ["Black", "White", "Navy", "Gray", "Red", "Blue", "Green", "Pink", "Yellow", "Purple"],
                default=["Black", "Navy"]
            )
            
            occasions = st.multiselect(
                "Common Occasions:",
                ["Work", "Casual Outings", "Parties", "Travel", "Sports", "Formal Events"],
                default=["Work", "Casual Outings"]
            )
        
        with col2:
            budget_range = st.selectbox(
                "Budget Range:",
                ["$0-50", "$50-100", "$100-200", "$200-500", "$500+"]
            )
            
            body_type = st.selectbox(
                "Body Type (Optional):",
                ["Not specified", "Petite", "Tall", "Plus Size", "Athletic", "Pear", "Apple", "Hourglass"]
            )
            
            style_inspiration = st.text_area(
                "Style Inspiration (Optional):",
                placeholder="Describe your style icons or inspiration..."
            )
        
        if st.button("Get Fashion Recommendations", type="primary"):
            user_preferences = {
                'styles': preferred_styles,
                'colors': preferred_colors,
                'occasions': occasions,
                'budget': budget_range,
                'body_type': body_type,
                'inspiration': style_inspiration
            }
            
            with st.spinner("Generating personalized recommendations..."):
                recommendations = assistant.get_fashion_recommendations(
                    user_preferences, 
                    st.session_state.get('style_history', [])
                )
                
                st.subheader("🎯 Your Personalized Fashion Recommendations")
                
                # Display recommendations in tabs
                tabs = st.tabs(["Outfit Ideas", "Shopping List", "Style Tips", "Trend Alerts"])
                
                with tabs[0]:
                    st.markdown("**Curated Outfit Ideas:**")
                    for i, outfit in enumerate(recommendations['outfits'], 1):
                        st.markdown(f"**{i}. {outfit['name']}**")
                        st.markdown(f"   *{outfit['description']}*")
                        st.markdown(f"   **Best for:** {outfit['occasion']}")
                        st.markdown("")
                
                with tabs[1]:
                    st.markdown("**Recommended Items to Shop:**")
                    for item in recommendations['shopping_list']:
                        st.markdown(f"• **{item['item']}** - {item['reason']}")
                
                with tabs[2]:
                    st.markdown("**Personalized Style Tips:**")
                    for tip in recommendations['style_tips']:
                        st.info(f"💡 {tip}")
                
                with tabs[3]:
                    st.markdown("**Trend Alerts for You:**")
                    for trend in recommendations['trend_alerts']:
                        st.success(f"🔥 {trend}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using PyTorch, CLIP, and Streamlit | Focus: Text-to-Image, Style Transfer, and Trend Analysis")

if __name__ == "__main__":
    main()