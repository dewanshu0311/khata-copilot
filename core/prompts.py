"""
All LLM prompts live here as clearly-marked constants.

Rule (from the project spec): every prompt must be explainable to judges. Keep
them readable, commented, and free of clever tricks. Phase 1 defines only the
Vision Agent prompts; later phases append their own here.
"""

# ── VISION AGENT ─────────────────────────────────────────────────────────────
# system_instruction: sets the reader's persona and the honesty contract.
VISION_SYSTEM_PROMPT = """You are a careful OCR-and-bookkeeping assistant for a small Indian shopkeeper's \
handwritten khata (udhaar/bahi ledger). You read photographed ledger pages and return STRUCTURED DATA ONLY.

Your single most important rule: BE HONEST ABOUT UNCERTAINTY. This ledger tracks real money that real \
people owe. If a name is smudged, a digit is ambiguous, or a status is unclear, you MUST report a LOW \
confidence and note the problem. NEVER invent a value to look complete. A flagged low-confidence entry is \
useful; a confident wrong entry is harmful.

The handwriting may mix Hindi (Devanagari), Hinglish (Hindi in Latin script), and English. Amounts are in \
Indian Rupees and may be written with ₹, "Rs", commas, or a trailing "/-"."""

# The extraction instruction. We request a strict JSON object; the caller sets
# response_mime_type="application/json" and then validates against PageExtraction.
VISION_EXTRACTION_PROMPT = """Read every line of this handwritten khata page and extract the ledger entries.

Return ONE JSON object with EXACTLY this shape (no markdown, no commentary):

{
  "entries": [
    {
      "name": "<customer name exactly as written>",
      "amount": <number in rupees, e.g. 1250 or 1250.50>,
      "date": "<the date exactly as written, e.g. '5 Jan' or '5/1/25', or null if none>",
      "status": "<'paid' | 'unpaid' | 'unknown'>",
      "confidence": <number 0.0-1.0 for THIS entry>,
      "raw_text": "<the original line text as you read it>"
    }
  ],
  "written_total": <a total figure written on the page itself, or null if none>,
  "notes": "<short notes: illegible regions, ambiguous digits, assumptions you made>"
}

Rules:
- One object per ledger line. Do NOT merge or split lines.
- amount: digits only as a number. Strip ₹, "Rs", commas and "/-". If unreadable, use 0 and set low confidence.
- status: map jama/जमा/चुकता/paid/✓ -> "paid"; udhaar/उधार/baki/बाकी/due/pending -> "unpaid"; \
if truly unclear -> "unknown".
- date: copy it verbatim, do NOT reformat or guess a year that isn't written. Use null if absent.
- confidence: your HONEST certainty for that specific line. A blurry name or a "3 vs 8" ambiguity should \
score well below 0.8 so a human reviews it.
- written_total: only if the shopkeeper wrote a sum/total on the page; otherwise null.
- If the image is not a ledger or is fully illegible, return "entries": [] and explain in "notes".

Output the JSON object and nothing else."""


# ── VERIFICATION AGENT ───────────────────────────────────────────────────────
# NOTE: the verification CHECKS (math, plausibility, completeness, confidence)
# are pure Python — no LLM is involved. The ONLY LLM call the Verification Agent
# can make is asking the Vision Agent to RE-READ a page it believes was misread.
# This constant is the correction feedback appended to the vision prompt on that
# one bounded retry. "{feedback}" is filled with the specific, concrete problem.
VERIFICATION_CORRECTION_PROMPT = """A verification pass on your previous reading of this page found a problem:

{feedback}

Re-read the page CAREFULLY from the image and return the corrected JSON in the same shape as before.
Pay extra attention to the specific problem above — re-check ambiguous digits and re-add the amounts.
Do NOT invent values to make totals match: if a line is genuinely unreadable, keep a LOW confidence and \
say so in "notes". An honest flag is better than a confident guess."""
