import base64
import sys
import os
import traceback
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QAData(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(payload: QAData):
    try:
        base64_str = payload.image_base64
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        base64_str = base64_str.strip().replace("\n", "").replace("\r", "")

        # Build strict system instruction rule
        system_instruction = (
            "You are a strict data extraction bot. Answer the question using ONLY the provided image. "
            "Rule: If the answer is a numeric value, return ONLY the raw number as a string. "
            "Do not include currency symbols, commas, units, letters, or extra spaces. "
            "Example: If the total is $4,089.35, reply exactly: 4089.35"
        )

        token = os.getenv("GEMINI_API_KEY")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Native Gemini layout for AI Pipe proxy endpoint
        json_data = {
            "contents": [
                {
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": base64_str}},
                        {"text": f"{system_instruction}\n\nQuestion: {payload.question}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0
            }
        }

        async with httpx.AsyncClient() as client:
            # We are using the native Gemini proxy routing on aipipe for maximum consistency
            response = await client.post(
                "https://aipipe.org/geminiv1beta/models/gemini-2.5-flash:generateContent",
                headers=headers,
                json=json_data,
                timeout=45.0
            )

            # Print status immediately to logs
            print(f"DIAGNOSTIC - Upstream Status: {response.status_code}", file=sys.stderr)

            if response.status_code != 200:
                print(f"DIAGNOSTIC - Upstream Error Body: {response.text}", file=sys.stderr)
                raise HTTPException(status_code=500, detail=f"AI Pipe Upstream Error: {response.text}")

            result = response.json()
            answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"answer": answer}

    except Exception as e:
        # CRITICAL: This extracts the actual line number and full error context to your Render Logs tab
        print("======== CRITICAL CODE EXCEPTION BREAKDOWN ========", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("===================================================", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
