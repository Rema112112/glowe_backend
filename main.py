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
      "productTypeName": "Specific Cleanser Type", 
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

        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.startswith("```"): raw_text = raw_text[3:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())

        # Normalize keys for both camelCase and snake_case compatibility
        prod_name = data.get("productName") or data.get("product_name") or "Unknown Product"
        score = data.get("compatibilityScore") or data.get("compatibility_score") or 85
        safe_sens = data.get("sensitiveSkinSafe") if "sensitiveSkinSafe" in data else data.get("sensitive_skin_safe", True)
        sens_note = data.get("sensitiveSkinNote") or data.get("sensitive_skin_note") or ""
        skin_types = data.get("suitableSkinTypes") or data.get("suitable_skin_types") or ["All Skin Types"]
        preg_safe = data.get("pregnancySafe") if "pregnancySafe" in data else data.get("pregnancy_safe", True)
        preg_note = data.get("pregnancyNote") or data.get("pregnancy_note") or ""
        med_caution = data.get("medicationCaution") if "medicationCaution" in data else data.get("medication_caution", False)
        med_note = data.get("medicationNote") or data.get("medication_note") or "None"
        ingredients = data.get("ingredients") or []

        return {
            "productName": prod_name, "product_name": prod_name,
            "compatibilityScore": score, "compatibility_score": score,
            "sensitiveSkinSafe": safe_sens, "sensitive_skin_safe": safe_sens,
            "sensitiveSkinNote": sens_note, "sensitive_skin_note": sens_note,
            "suitableSkinTypes": skin_types, "suitable_skin_types": skin_types,
            "pregnancySafe": preg_safe, "pregnancy_safe": preg_safe,
            "pregnancyNote": preg_note, "pregnancy_note": preg_note,
            "medicationCaution": med_caution, "medication_caution": med_caution,
            "medicationNote": med_note, "medication_note": med_note,
            "ingredients": ingredients
        }
        
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
            max_completion_tokens=1500,
        )

        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.startswith("```"): raw_text = raw_text[3:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())

        # Normalize keys for both camelCase and snake_case compatibility
        explanation = data.get("skinAnalysisExplanation") or data.get("skin_analysis_explanation") or data.get("explanation") or "Skin analysis completed successfully."
        conditions = data.get("detectedConditions") or data.get("detected_conditions") or data.get("conditions") or {}
        routine = data.get("routineRecommendations") or data.get("routine_recommendations") or data.get("recommendations") or data.get("routine") or {}

        return {
            "skinAnalysisExplanation": explanation,
            "skin_analysis_explanation": explanation,
            "explanation": explanation,
            
            "detectedConditions": conditions,
            "detected_conditions": conditions,
            "conditions": conditions,
            
            "routineRecommendations": routine,
            "routine_recommendations": routine,
            "recommendations": routine,
            "routine": routine
        }
        
    except Exception as e:
        print(f"Error in /analyze-skin: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
