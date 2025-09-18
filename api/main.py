from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.text_to_fashion import TextToFashionGenerator
from models.style_transfer import StyleTransferModel
from models.virtual_tryon import VirtualTryOnModel
from models.trend_predictor import TrendPredictor

app = FastAPI(title="AI Fashion Design Assistant API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
text2fashion = TextToFashionGenerator()
style_transfer = StyleTransferModel()
virtual_tryon = VirtualTryOnModel()
trend_predictor = TrendPredictor()

@app.get("/")
async def root():
    return {"message": "AI Fashion Design Assistant API"}

@app.post("/generate-design/")
async def generate_design(prompt: str, style: str = "modern"):
    try:
        image = text2fashion.generate(prompt, style)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(io.BytesIO(img_byte_arr.read()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/style-transfer/")
async def apply_style_transfer(content: UploadFile = File(...), style: UploadFile = File(...)):
    try:
        content_image = Image.open(io.BytesIO(await content.read()))
        style_image = Image.open(io.BytesIO(await style.read()))
        
        result = style_transfer.transfer(content_image, style_image)
        
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(io.BytesIO(img_byte_arr.read()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtual-tryon/")
async def virtual_try_on(person: UploadFile = File(...), clothing: UploadFile = File(...)):
    try:
        person_image = Image.open(io.BytesIO(await person.read()))
        clothing_image = Image.open(io.BytesIO(await clothing.read()))
        
        result = virtual_tryon.try_on(person_image, clothing_image)
        
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(io.BytesIO(img_byte_arr.read()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-trends/")
async def predict_trends(images: list[UploadFile] = File(...)):
    try:
        image_list = []
        for img_file in images:
            image = Image.open(io.BytesIO(await img_file.read()))
            image_list.append(image)
        
        predictions = trend_predictor.predict(image_list)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)