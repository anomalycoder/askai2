import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Invoice Extraction API")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
        response = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract invoice information. "
                        "Return ONLY valid JSON matching the supplied schema."
                    ),
                },
                {
                    "role": "user",
                    "content": req.text,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice",
                    "strict": True,
                    "schema": req.schema,
                },
            },
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
