import pytest
import torch
from PIL import Image
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.text_to_fashion import TextToFashionGenerator
from models.style_transfer import StyleTransferModel
# from models.virtual_tryon import VirtualTryOnModel
from models.trend_predictor import TrendPredictor

def create_dummy_image(size=(256, 256), color=(255, 255, 255)):
    """Create a dummy image for testing"""
    return Image.new('RGB', size, color)

def test_text_to_fashion():
    model = TextToFashionGenerator()
    result = model.generate("red dress", "modern")
    assert isinstance(result, Image.Image)
    assert result.size == (256, 256)

def test_style_transfer():
    model = StyleTransferModel()
    content_img = create_dummy_image(color=(100, 100, 100))
    style_img = create_dummy_image(color=(200, 50, 50))
    
    result = model.transfer(content_img, style_img)
    assert isinstance(result, Image.Image)
    assert result.size == content_img.size

# def test_virtual_tryon():
#     model = VirtualTryOnModel()
#     person_img = create_dummy_image(color=(150, 100, 80))
#     clothing_img = create_dummy_image(color=(0, 0, 255))
    
#     result = model.try_on(person_img, clothing_img)
#     assert isinstance(result, Image.Image)
#     assert result.size == person_img.size

def test_trend_predictor():
    model = TrendPredictor()
    images = [create_dummy_image() for _ in range(3)]
    
    result = model.predict(images)
    assert isinstance(result, dict)
    assert 'colors' in result
    assert 'styles' in result
    assert 'patterns' in result
    assert 'season' in result

if __name__ == "__main__":
    pytest.main([__file__])