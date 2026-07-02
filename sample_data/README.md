# sample_data

Drop real khata (handwritten ledger) photos here — `.jpg` / `.png`.

Then run, from the repo root:

```bash
python scripts/test_vision.py sample_data/your_page.jpg
```

No photo or API key yet? The vision agent falls back to **mock mode** (a canned,
clearly-flagged extraction) so the CLI and later the UI still run:

```bash
python scripts/test_vision.py           # uses mock mode automatically
# or force it explicitly:
KHATA_MOCK=1 python scripts/test_vision.py
```

Good test images for a hackathon demo: a page with a written total (so the
Phase 2 Verification Agent can check the math), and at least one smudged/ambiguous
line (so you can show low-confidence entries getting flagged for review).
