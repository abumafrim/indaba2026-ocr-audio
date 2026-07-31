# From Archive to AI: OCR & Speech — Deep Learning Indaba, Lagos 2026

A 30-minute live demonstration for legal, media and NLP practitioners working on **African data discoverability**. One message, shown twice:

> **Data is not yet a dataset.** Between the archive and the AI model there is human work that adds value. This demo shows that work.

| # | Notebook | Story | Open in Colab |
|---|----------|-------|---------------|
| 1 | [`01_ocr_scan_to_text.ipynb`](notebooks/01_ocr_scan_to_text.ipynb) | An aged Hausa newspaper page becomes machine-readable text (about 60% → under 2% character error) — but the letters ƙ ɗ ɓ are lost, because Tesseract has **no Hausa model** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abumafrim/indaba2026-ocr-audio/blob/main/notebooks/01_ocr_scan_to_text.ipynb) |
| 2 | [`02_audio_tape_to_corpus.ipynb`](notebooks/02_audio_tape_to_corpus.ipynb) | A noisy 48 kHz field tape of real Hausa speech becomes 10 clean, labelled clips plus a manifest — a small corpus | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abumafrim/indaba2026-ocr-audio/blob/main/notebooks/02_audio_tape_to_corpus.ipynb) |

> **After pushing to GitHub:** if the repo lives somewhere other than `abumafrim/indaba2026-ocr-audio`, update the badge URLs above and the `REPO_URL` variable in each notebook's Setup cell.

Each notebook is a **12-minute demo** followed by a clearly separated **self-study appendix** (smarter binarization, layout analysis, neural OCR, Whisper/VAD, corpus QC, dataset pointers) for attendees who go further afterwards.

## Session context

The demo slots between a stakeholder round table on *unusual sources of datasets* and a talk on *IP, copyright and licensing*. Both notebooks end by handing off to that IP conversation: the value in a dataset was put there by people — speakers, transcribers, annotators, engineers — and that value has owners.

## Repository layout

```
notebooks/
  01_ocr_scan_to_text.ipynb            the live OCR demo
  02_audio_tape_to_corpus.ipynb        the live audio demo
  *_EXECUTED_BACKUP.ipynb              same notebooks with all outputs saved —
                                       open these if wifi/Colab fails on the day
data/
  newspaper_raw.png                    the "1972 Hausa newspaper" scan (synthetic, aged)
  newspaper_clean.png / *_ground_truth.txt
  yoruba_snippet.png                   Yoruba sentence for the diacritics contrast
  field_recording_raw.wav              22.8 s simulated field tape (real Hausa speech)
  word_list.txt                        the 10 words, in spoken order
  source_words/                        original CC0 recordings + CREDITS.txt
  tessdata/                            OCR models the notebooks use via a relative
                                       path (eng + yor, Apache-2.0, tesseract-ocr)
scripts/
  make_ocr_assets.py                   renders + ages the newspaper (so ground truth is known)
  make_audio_assets.py                 splices + degrades the field tape
  build_notebooks.py                   regenerates both notebooks
```

## Reproducing / editing

```bash
pip install pillow numpy opencv-python pytesseract jiwer librosa soundfile noisereduce nbformat
python3 scripts/make_ocr_assets.py      # rebuild the newspaper assets
python3 scripts/make_audio_assets.py    # rebuild the field tape
python3 scripts/build_notebooks.py      # rebuild both notebooks
```

The assets are deliberately synthetic-but-honest: because we rendered the page and spliced the tape ourselves, the ground truth is known exactly and every error rate shown live is real. The appendices disclose this to attendees — with real archives, ground truth itself is human work, which is the point of the session.

## Data credits

- Hausa word recordings: Wikimedia Commons, [Hausa pronunciation](https://commons.wikimedia.org/wiki/Category:Hausa_pronunciation) category, recorded by **Gwanki**, CC0. Full list in [`data/source_words/CREDITS.txt`](data/source_words/CREDITS.txt).
- `yor.traineddata`: [tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata), Apache-2.0; `eng.traineddata`: [tesseract-ocr/tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast), Apache-2.0.
- Newspaper page: synthetic Hausa text written for this workshop, aged programmatically.
