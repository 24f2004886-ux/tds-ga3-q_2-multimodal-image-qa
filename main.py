import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# -------------------------------------------------------------
# REQUIREMENT: Enable CORS so the external grader can access it
# -------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin (like the grader)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini Client
# It automatically looks for an environment variable named GEMINI_API_KEY
client = genai.Client()

# Define what the incoming request data must look like
class QA_Request(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(payload: QA_Request):
    try:
        # 1. Clean the base64 string if it contains a data URI prefix
        b64_string = payload.image_base64
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        # 2. Decode the base64 string back into raw image bytes
        image_bytes = base64.b64decode(b64_string)

        # 3. Format the image bytes using Google's Part.from_bytes helper
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )

        # 4. Craft a strict system instruction to enforce grading requirements
        system_instruction = (
            "You are a strict data extraction bot. Answer the question using ONLY the provided image. "
            "Rule: If the answer is a numeric value, return ONLY the raw number as a string. "
            "Do not include currency symbols, commas, units, letters, or extra spaces. "
            "Example: If the total is $4,089.35, reply exactly: 4089.35"
        )

        # 5. Call the Gemini 2.5 Flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, payload.question],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0 # Lowest randomness for maximum accuracy
            )
        )

        # 6. Extract and clean the final text response
        final_answer = response.text.strip()

        # RETURN FORMAT REQUIREMENT: {"answer": "..."}
        return {"answer": final_answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
