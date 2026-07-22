import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types

app = FastAPI(title="Invoice Extraction API")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class ExtractRequest(BaseModel):
    document_id: str
    text: str
    schema: dict


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/extract")
def extract(req: ExtractRequest):
    try:

        prompt = f"""
You are an invoice extraction engine.

Extract all fields from the invoice.

IMPORTANT:
- Return ONLY JSON.
- Follow the JSON Schema exactly.
- Do not include markdown.
- Do not explain anything.

JSON Schema:

{json.dumps(req.schema, indent=2)}

Invoice Text:

{req.text}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
