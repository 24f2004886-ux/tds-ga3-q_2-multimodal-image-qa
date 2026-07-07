import base64
import sys
import os
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

        # Clean string spaces or newlines that can break HTTP transfer
        base64_str = base64_str.strip().replace("\n", "").replace("\r", "")

        # Re-verify correct padding format
        base64_str += "=" * ((4 - len(base64_str) % 4) % 4)

        # Build standard data URI string
        image_url_data = f"data:image/png;base64,{base64_str}"

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

        # Structuring multimodal payload for OpenRouter spec via AI Pipe
        json_data = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{system_instruction}\n\nQuestion: {payload.question}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url_data
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://aipipe.org/openrouter/v1/chat/completions",
                headers=headers,
                json=json_data,
                timeout=45.0
            )

            if response.status_code != 200:
                print(f"AI Pipe Error Response: {response.text}", file=sys.stderr)
                raise HTTPException(status_code=500, detail=f"Upstream API failure: {response.text}")

            result = response.json()

            if "choices" not in result or not result["choices"]:
                print(f"Unexpected JSON format from API: {result}", file=sys.stderr)
                raise HTTPException(status_code=500, detail="Empty response structure from AI model.")

            answer = result["choices"][0]["message"]["content"].strip()
            return {"answer": answer}

    except Exception as e:
        print(f"CRITICAL API EXCEPTION: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
