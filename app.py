import json
import re
from datetime import datetime
from typing import Dict, Any

from dateutil.parser import parse
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from word2number import w2n

client = OpenAI()

app = FastAPI()


# --------------------------
# Request Schema
# --------------------------

class InvoiceRequest(BaseModel):
    document_id: str
    text: str
    schema: Dict[str, Any]


# --------------------------
# Helpers
# --------------------------

CURRENCY_MAP = {
    "$": "USD",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",

    "€": "EUR",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",

    "£": "GBP",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "pounds sterling": "GBP",

    "₹": "INR",
    "rs": "INR",
    "inr": "INR",
    "rupees": "INR",

    "¥": "JPY",
    "jpy": "JPY",
    "yen": "JPY",
}


def normalize_currency(value: str):
    if value is None:
        return value

    value = value.lower().strip()

    for k, v in CURRENCY_MAP.items():
        if k in value:
            return v

    return value.upper()


def parse_integer(value):
    """
    Handles

    12,480
    1,24,800
    twelve thousand four hundred
    12K
    4M
    """

    if isinstance(value, int):
        return value

    if value is None:
        return None

    s = str(value).strip()

    s = s.replace(",", "")

    m = re.match(r"([\d\.]+)\s*([kKmM])$", s)

    if m:
        num = float(m.group(1))
        suf = m.group(2).lower()

        if suf == "k":
            return int(num * 1000)

        return int(num * 1000000)

    if re.fullmatch(r"\d+", s):
        return int(s)

    try:
        return w2n.word_to_num(s)
    except:
        return value


def normalize_date(date_string):
    if date_string is None:
        return None

    return parse(date_string, dayfirst=False).strftime("%Y-%m-%d")


def normalize_due(value):
    """
    Handles:

    Net 30

    payable within 45 days

    due in two weeks
    """

    if isinstance(value, int):
        return value

    s = str(value).lower()

    m = re.search(r'(\d+)\s*day', s)

    if m:
        return int(m.group(1))

    m = re.search(r'net\s*(\d+)', s)

    if m:
        return int(m.group(1))

    m = re.search(r'(\d+)\s*week', s)

    if m:
        return int(m.group(1)) * 7

    try:
        if "week" in s:
            words = s.replace("weeks", "").replace("week", "")
            return w2n.word_to_num(words) * 7
    except:
        pass

    return value


EXPECTED_KEYS = [
    "vendor",
    "currency",
    "total_amount",
    "invoice_date",
    "due_in_days",
    "is_paid",
    "priority",
    "contact_email",
    "line_items",
    "item_count",
]


# --------------------------
# API
# --------------------------

@app.post("/extract")
def extract(req: InvoiceRequest):

    prompt = f"""
Extract invoice information.

Return ONLY valid JSON.

Document:

{req.text}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "invoice",
                "schema": req.schema,
                "strict": True,
            }
        },
    )

    result = json.loads(response.output_text)

    # -----------------------
    # Normalization
    # -----------------------

    result["currency"] = normalize_currency(result["currency"])

    result["total_amount"] = parse_integer(result["total_amount"])

    result["invoice_date"] = normalize_date(result["invoice_date"])

    result["due_in_days"] = normalize_due(result["due_in_days"])

    result["contact_email"] = result["contact_email"].lower()

    for item in result["line_items"]:
        item["quantity"] = parse_integer(item["quantity"])
        item["unit_price"] = parse_integer(item["unit_price"])

    result["item_count"] = len(result["line_items"])

    # -----------------------
    # Validate exact keys
    # -----------------------

    if set(result.keys()) != set(EXPECTED_KEYS):
        raise HTTPException(
            status_code=400,
            detail="Returned JSON keys do not match schema."
        )

    return {k: result[k] for k in EXPECTED_KEYS}
