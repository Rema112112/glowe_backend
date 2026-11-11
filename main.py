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

# Retrieve Groq API Key from environment variables
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    raise ValueError("❌ GROQ_API_KEY is missing! Check your Railway Variables tab.")

# Initialize OpenAI client configured for Groq's LPU endpoint
client = OpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1"
)

# Groq multimodal vision model
MODEL_NAME = "qwen/qwen3.8-27b"

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
You MUST provide a comprehensive explanation of the skin condition (noting if it's oily, dry, showing rashes, etc.) AND fully populate a tailored 5-step routine with specific product types, reasons, and active key ingredients for: Cleanser, Toner, Serum, Moisturizer, and Sunscreen.

Return ONLY a valid JSON object matching this exact schema, with no markdown code blocks or extra text:
{
  "skinAnalysisExplanation": "Detailed explanation of skin condition, noting dryness, oiliness, rashes or irritation.",
  "detectedConditions": {
    "acne": "Status or Severity",
    "redness": "Status or Severity",
    "dryness": "Status or Severity",
    "pores": "Status or Severity",
    "hyperpigmentation": "Status or Severity",
    "texture": "Status or Severity"
  },
  "routineRecommendations": {
    "cleanser": {
      "productTypeName": "Specific Cleanser Type (e.g., Gentle Hydrating Cleanser)", 
      "reason": "Why this is chosen based on detected conditions", 
      "keyIngredients": ["Ceramides", "Glycerin"]
    },
    "toner": {
      "productTypeName": "Specific Toner Type", 
      "reason": "Why this is chosen", 
      "keyIngredients": ["Ingredient 1"]
    },
    "serum": {
      "productTypeName": "Specific Serum Type", 
      "reason": "Why this is chosen", 
      "keyIngredients": ["Ingredient 1"]
    },
    "moisturizer": {
      "productTypeName": "Specific Moisturizer Type", 
      "reason": "Why this is chosen", 
      "keyIngredients": ["Ingredient 1"]
    },
    "sunscreen": {
      "productTypeName": "Specific Sunscreen Type", 
      "reason": "Why this is chosen", 
      "keyIngredients": ["Zinc Oxide"]
    }
  }
}
"""

@app.get("/")
def root():
    return {"message": "Glowe Groq backend is running"}

@app.post("/analyze")
def analyze_text(request: PromptRequest):
    try:
        user_input = request.prompt.strip()
        
        if user_input.isdigit():
            user_prompt = f"Barcode EAN/UPC: {user_input}. Identify this exact cosmetic product."
        else:
            user_prompt = user_input

        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_completion_tokens=1000,
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SKIN_SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this skin image, identify dryness/oiliness/rashes, and generate the full evaluation and routine recommendation JSON."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_completion_tokens=1500,  # Increased token limit so routine fields don't cut off
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in /analyze-skin: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
