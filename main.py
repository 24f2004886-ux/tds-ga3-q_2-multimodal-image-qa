import base64
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

class QA_Request(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(payload: QA_Request):
    try:
        b64_string = payload.image_base64
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        # Fix any potential base64 padding issues automatically
        b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
        image_bytes = base64.b64decode(b64_string)

        # Dynamically determine if the image is PNG or JPEG based on magic numbers
        mime_type = "image/png"
        if image_bytes.startswith(b"\xff\xd8"):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b"\x89PNG"):
            mime_type = "image/png"
        elif image_bytes.startswith(b"GIF8"):
            mime_type = "image/gif"

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        system_instruction = (
            "You are a strict data extraction bot. Answer the question using ONLY the provided image. "
            "Rule: If the answer is a numeric value, return ONLY the raw number as a string. "
            "Do not include currency symbols, commas, units, letters, or extra spaces. "
            "Example: If the total is $4,089.35, reply exactly: 4089.35"
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, payload.question],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            )
        )

        final_answer = response.text.strip()
        return {"answer": final_answer}

    except Exception as e:
        # CRITICAL: This prints the exact error back into your Render logs
        print(r"--- ERROR ENCOUNTERED ---", file=sys.stderr)
        print(str(e), file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
