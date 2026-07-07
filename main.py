import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. Initialize the FastAPI app
app = FastAPI()

# 2. Enable CORS (Required so the assignment grader can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 3. Securely load your Gemini API Key
# Make sure to set this environment variable, or paste your key string directly for local testing:
# client = genai.Client(api_key="YOUR_ACTUAL_API_KEY")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 4. Define what incoming data should look like
class QAData(BaseModel):
    image_base64: str
    question: str

# 5. Create the required assignment endpoint
@app.post("/answer-image")
def answer_image(data: QAData):
    try:
        # Clean the base64 string if it contains metadata prefixes
        base64_str = data.image_base64
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # Decode the text string back into raw image bytes
        image_bytes = base64.b64decode(base64_str)

        # Prepare the image for Gemini
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",  # Adjust if handling purely JPEGs
        )

        # Enforce the assignment formatting rule using a system prompt
        system_instruction = (
            "You are a precise data extraction assistant. "
            "If the answer is a number, return ONLY the raw numeric value. "
            "Do not include currency symbols ($), commas, or units (kg, items). "
            "Keep answers as brief and direct as possible."
        )

        # Call the multimodal Gemini model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, data.question],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1 # Low temperature ensures strict factual extraction
            )
        )

        # Return the response exactly as required by the spec
        return {"answer": response.text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally using: uvicorn main:app --reload
