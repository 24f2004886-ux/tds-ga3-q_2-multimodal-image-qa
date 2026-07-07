import base64
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Automatically loads GEMINI_API_KEY from Render's Environment Variables
client = genai.Client()

class QAData(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(data: QAData):  # Re-added 'async' to support multi-concurrency grading requests
    try:
        base64_str = data.image_base64
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # Automatically fix any string padding issues if the grader truncates trailing '=' signs
        base64_str += "=" * ((4 - len(base64_str) % 4) % 4)
        image_bytes = base64.b64decode(base64_str)

        # Dynamically detect image type from its raw file signature header
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

        system_instruction = (
            "You are a strict data extraction bot. Answer the question using ONLY the provided image. "
            "Rule: If the answer is a numeric value, return ONLY the raw number as a string. "
            "Do not include currency symbols, commas, units, letters, or extra spaces. "
            "Example: If the total is $4,089.35, reply exactly: 4089.35"
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, data.question],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            )
        )

        return {"answer": response.text.strip()}

    except Exception as e:
        # In case anything fails, this prints the exact traceback directly to your Render Logs tab
        print(f"CRITICAL API ERROR: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
