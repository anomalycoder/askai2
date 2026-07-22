import json
import os

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
    prompt = f"""
You are an expert invoice extraction engine.

Extract information from the invoice.

Rules:
- Return ONLY valid JSON.
- Follow the JSON Schema exactly.
- Do not add extra keys.
- Preserve vendor exactly as written.
- Currency must be ISO4217 (USD, EUR, GBP, INR, JPY).
- total_amount must be an integer.
- invoice_date must be YYYY-MM-DD.
- due_in_days must be integer.
- is_paid must be boolean.
- priority must be one of:
  low
  normal
  high
  urgent
- contact_email must be lowercase.
- unit_price must be integer.
- quantity must be integer.
- item_count must equal len(line_items).

Schema:

{json.dumps(req.schema, indent=2)}

Invoice:

{req.text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=req.schema,
                temperature=0,
            ),
        )

        data = json.loads(response.text)

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
