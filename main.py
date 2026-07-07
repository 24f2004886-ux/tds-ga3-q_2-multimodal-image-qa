import base64
import sys
import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS (Required so the automated Cloudflare grader can reach it)
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

        # Structure the payload using AI Pipe's standard OpenRouter formatting spec
        # Vision models accept the image data embedded directly into a Data URI
        image_url = f"data:image/png;base64,{base64_str}"

        system_instruction = (
            "You are a precise data extraction assistant. "
            "Answer the question using ONLY the provided image. "
            "If the answer is a number, return ONLY the raw numeric value as a string. "
            "Do not include currency symbols ($), commas, spaces, or units. "
            "Example: If the total is $4,089.35, reply exactly: 4089.35"
        )

        # Get the token from your environment variables setup on Render
        token = os.getenv("GEMINI_API_KEY")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Request payload targeting the standard course-provided models
        json_data = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": payload.question},
                        {"type": "image_url", "image_url": {"url": image_url}}
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
                timeout=30.0
            )

            if response.status_code != 200:
                print(f"AI Pipe Error Response: {response.text}", file=sys.stderr)
                raise HTTPException(status_code=500, detail=f"AI Pipe Error: {response.text}")

            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            return {"answer": answer}

    except Exception as e:
        print(f"CRITICAL API EXCEPTION: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
