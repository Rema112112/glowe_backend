import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Only load .env file if it exists locally (prevents overriding Railway variables in production)
if os.path.exists(".env"):
    load_dotenv()

app = FastAPI()

# Retrieve and validate the API key
gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    raise ValueError("❌ GEMINI_API_KEY is missing! Check your Railway Variables tab.")

client = genai.Client(api_key=gemini_key)

# Fixed: Use a valid, stable, and fast model name
MODEL_NAME = "gemini-2.5-flash"

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
    return {"message": "Glowe backend is running"}

@app.post("/analyze")
def analyze_text(request: PromptRequest):
    try:
        user_input = request.prompt.strip()
        
        if user_input.isdigit():
            full_prompt = f"{SYSTEM_INSTRUCTION}\n\nUser Input (Barcode EAN/UPC): {user_input}. Identify this exact cosmetic product."
        else:
            full_prompt = f"{SYSTEM_INSTRUCTION}\n\nUser Input: {user_input}"
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=500
            )
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.startswith("```"): raw_text = raw_text[3:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
        
    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        # Fallback MIME type if it's application/octet-stream or missing
        mime_type = file.content_type
        if not mime_type or mime_type == "application/octet-stream":
            mime_type = "image/jpeg"
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                SKIN_SYSTEM_INSTRUCTION
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=500
            )
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.startswith("```"): raw_text = raw_text[3:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
        
    except Exception as e:
        print(f"Error in /analyze-skin: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
