import base64
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS (Required so the Cloudflare Worker grader can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Automatically pulls the GEMINI_API_KEY from your Render Environment Variables
client = genai.Client()

# This schema exactly matches the spec from your q-multimodal-image-qa-server_sample.json
class QAData(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(payload: QAData): # Changed from 'data' to 'payload' to fix the internal 500 error
    try:
        base64_str = payload.image_base64

        # Clean data URI prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # Fix any potential base64 padding issues automatically
        base64_str += "=" * ((4 - len(base64_str) % 4) % 4)
        image_bytes = base64.b64decode(base64_str)

        # Dynamically determine the MIME type based on raw image bytes
        mime_type = "image/png"
        if image_bytes.startswith(b"\xff\xd8"):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b"\x89PNG"):
            mime_type = "image/png"
        elif image_bytes.startswith(b"GIF8"):
            mime_type = "image/gif"

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        # Enforce strict assignment formatting rules
        system_instruction = (
            "You are a precise data extraction assistant. "
            "Answer the question using ONLY the provided image. "
            "If the answer is a number, return ONLY the raw numeric value as a string. "
            "Do not include currency symbols ($), commas, spaces, or units (kg, items). "
            "Example: If the total is $4,089.35, reply exactly: 4089.35"
        )

        # Call the multimodal Gemini model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, payload.question],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0 # Lowest randomness for strict factual extraction
            )
        )

        # Return the response structure exactly required by the spec: {"answer": "..."}
        return {"answer": response.text.strip()}

    except Exception as e:
        # Logs errors explicitly into the Render dashboard text console
        print(f"CRITICAL ERROR ENCOUNTERED: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
