import os
import json
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

if os.path.exists(".env"):
    load_dotenv()

app = FastAPI()

# Retrieve and validate DeepSeek API key
deepseek_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_key:
    raise ValueError("❌ DEEPSEEK_API_KEY is missing! Check your Railway Variables tab.")

# Initialize OpenAI client pointed at DeepSeek's endpoint
client = OpenAI(
    api_key=deepseek_key,
    base_url="https://api.deepseek.com"
)

MODEL_NAME = "deepseek-flash"

class PromptRequest(BaseModel):
    prompt: str

SYSTEM_INSTRUCTION = """
Analyze these skincare ingredients, product name, or barcode number.
Return ONLY a valid JSON object matching this exact schema, with no markdown code blocks or extra text:
{
  "productName": "Recognized Product Name",
  "compatibilityScore": 88,
  "sensitiveSkinSafe": true,
  "sensitiveSkinNote": "Detailed explanation",
  "suitableSkinTypes": ["Dry", "Normal", "Sensitive"],
  "pregnancySafe": true,
  "pregnancyNote": "Detailed explanation",
  "medicationCaution": false,
  "medicationNote": "None",
  "ingredients": [
    {
      "name": "Ingredient Name",
      "effect": "Detailed skin effect",
      "status": "Safe"
    }
  ]
}
Note: 'status' MUST be exactly one of: 'Safe', 'Caution', or 'Harmful'.
'compatibilityScore' must be an integer between 0 and 100.
"""

SKIN_SYSTEM_INSTRUCTION = """
You are an expert dermatologist. Analyze the user's face image and evaluate 6 conditions: Acne, Redness, Dryness, Pores, Hyperpigmentation, Texture.
Provide an explanation of the skin condition, and a 5-step routine: Cleanser, Toner, Serum, Moisturizer, Sunscreen.

Return ONLY a valid JSON object matching this exact schema, with no markdown code blocks or extra text:
{
  "skinAnalysisExplanation": "Detailed explanation of skin condition.",
  "detectedConditions": {
    "acne": "Status",
    "redness": "Status",
    "dryness": "Status",
    "pores": "Status",
    "hyperpigmentation": "Status",
    "texture": "Status"
  },
  "routineRecommendations": {
    "cleanser": {"productTypeName": "Type", "reason": "Reason", "keyIngredients": ["Ing1"]},
    "toner": {"productTypeName": "Type", "reason": "Reason", "keyIngredients": ["Ing1"]},
    "serum": {"productTypeName": "Type", "reason": "Reason", "keyIngredients": ["Ing1"]},
    "moisturizer": {"productTypeName": "Type", "reason": "Reason", "keyIngredients": ["Ing1"]},
    "sunscreen": {"productTypeName": "Type", "reason": "Reason", "keyIngredients": ["Ing1"]}
  }
}
"""

@app.get("/")
def root():
    return {"message": "Glowe DeepSeek backend is running"}

@app.post("/analyze")
def analyze_text(request: PromptRequest):
    try:
        user_input = request.prompt.strip()
        
        if user_input.isdigit():
            user_msg = f"Barcode EAN/UPC: {user_input}. Identify this exact cosmetic product."
        else:
            user_msg = user_input
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_text = response.choices[0].message.content.strip()
        return json.loads(raw_text)
        
    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        
        # Convert image bytes to base64 data URL for DeepSeek vision
        b64_encoded = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:{mime_type};base64,{b64_encoded}"
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SKIN_SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this user skin image and provide the JSON evaluation."},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        
        raw_text = response.choices[0].message.content.strip()
        return json.loads(raw_text)
        
    except Exception as e:
        print(f"Error in /analyze-skin: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
