"""Build the two workshop notebooks programmatically.

Regenerate with:  python3 scripts/build_notebooks.py
Outputs: notebooks/01_ocr_scan_to_text.ipynb, notebooks/02_audio_tape_to_corpus.ipynb

Design notes
  - Setup cells use Colab's `#@title ... { display-mode: "form" }` so they render
    collapsed but expandable (double-click to read the code).
  - Demo cells stay short, open and readable — the code is part of the show.
  - Audience-facing text is written in plain English for an audience that speaks
    English as a second or third language: short sentences, common words, no idioms.
  - Every number printed live was verified against this repo's data before the
    notebooks were written.
"""
import os
import re

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "notebooks")

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def unwrap_markdown(text):
    """Join hard-wrapped paragraph lines into single lines (renders the same)."""
    out, para, quote, fence = [], [], [], False

    def flush():
        if para:
            out.append(" ".join(para))
            para.clear()
        if quote:
            out.append("> " + " ".join(quote))
            quote.clear()

    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            flush()
            fence = not fence
            out.append(line)
            continue
        if fence:
            out.append(line)
            continue
        if s.startswith("> "):
            if para:
                flush()
            quote.append(s[2:])
            continue
        if not s or s.startswith(("#", "|", "---")):
            flush()
            out.append(line)
        elif s.startswith(("- ", "* ")) or re.match(r"\d+\.\s", s):
            flush()
            para.append(s)          # list item: continuation lines join onto it
        else:
            if quote:
                flush()
            para.append(s)
    flush()
    return "\n".join(out)


def save(nb, name):
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = unwrap_markdown(cell.source)
        cell.source = cell.source.rstrip("\n")
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": []}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    nbf.write(nb, path)
    print("wrote", path)


# =====================================================================
# Notebook 1 — OCR
# =====================================================================

nb1 = nbf.v4.new_notebook()
C = nb1.cells

C.append(md("""\
# From Scanned Page to Machine-Readable Text
### *Daga takarda zuwa data* — Deep Learning Indaba, Lagos 2026

Much of Africa's written record exists only on paper: newspapers, manuscripts,
court records, school and church registers. In this demo we take one old newspaper
page and follow the steps that turn it into text a computer can use.

The question for the next 12 minutes:

> *"We have the archives. Is that not enough?"*

The plan:
1. **The asset** — what an archive really holds
2. **The work** — the steps that turn a scan into text
3. **The result, and the gap** — what we recovered, and what is still missing

Note: cells with a grey title contain setup code. They are collapsed to keep the
page short. Double-click any of them to read the code inside. Every other cell
runs in front of you.
"""))

C.append(code("""\
#@title Setup — installs and data (double-click to read the code) { display-mode: "form" }
# Installs the Tesseract OCR engine and Python helpers, downloads the workshop
# data, and defines two small utilities.
import os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run(["apt-get", "install", "-y", "-q", "tesseract-ocr"], capture_output=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytesseract", "jiwer"],
                   capture_output=True)

REPO_URL = "https://github.com/abumafrim/indaba2026-ocr-audio"  # update if you fork
if os.path.exists("../data"):
    ROOT = ".."
elif os.path.exists("data"):
    ROOT = "."
else:
    ROOT = "indaba2026-ocr-audio"
    if not os.path.exists(os.path.join(ROOT, "data")):   # not downloaded yet
        result = subprocess.run(["git", "clone", "-q", REPO_URL + ".git", ROOT],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Could not download the workshop data. Git said:\n"
                               + result.stderr.strip())
DATA = os.path.join(ROOT, "data")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from jiwer import cer

# The repo ships its own OCR models (data/tessdata: eng + yor), so Tesseract
# reads them from a relative path — no system folders involved.
os.environ["TESSDATA_PREFIX"] = os.path.abspath(os.path.join(DATA, "tessdata"))

def show(*imgs, titles=None, height=5.5):
    \"\"\"Display 1-3 images side by side.\"\"\"
    fig, axes = plt.subplots(1, len(imgs), figsize=(7.5 * len(imgs), height))
    for ax, im, t in zip(np.atleast_1d(axes), imgs, (titles or [""] * len(imgs))):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=13)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def CER(truth_text, ocr_text):
    \"\"\"Character error rate, in % — how far the OCR text is from the true text.\"\"\"
    squash = lambda s: " ".join(s.split())
    return round(cer(squash(truth_text), squash(ocr_text)) * 100, 1)

truth = open(os.path.join(DATA, "newspaper_ground_truth.txt")).read()
print("Setup complete.  Tesseract", pytesseract.get_tesseract_version())
"""))

C.append(md("""\
## Part 1 — The asset

Text reaches a computer in three main ways. It was typed on a computer from the
start. Or someone typed it up later. Or it was scanned and converted. For most
African-language text from before the 2000s, scanning is the only option.

Here is our page: a 1972 Hausa newspaper (made for this demo), the way it comes
off a scanner in a media-house archive. It is tilted, it is stained, and the back
page shows through the paper.
"""))

C.append(code("""\
scan = cv2.imread(os.path.join(DATA, "newspaper_raw.png"), cv2.IMREAD_GRAYSCALE)
show(scan, titles=[f"One scanned page  ({scan.shape[1]}×{scan.shape[0]} pixels)"], height=8)
"""))

C.append(md("""\
A person can read this page. Can a machine? Let us ask Tesseract, the most widely
used free OCR engine, to read the page exactly as it was scanned.
"""))

C.append(code("""\
raw_text = pytesseract.image_to_string(scan, lang="eng")

print(raw_text[:400])
print("…")
print(f"\\nCharacter error rate: {CER(truth, raw_text)} % of the characters are wrong")
"""))

C.append(md("""\
About 60% of the characters are wrong or out of place. Look closely and you will
see two different failures. First, some characters are misread. Second, the tilt
confused the engine about the two columns, so it wove them together: the output
jumps from column one into column two in the middle of a sentence. Shuffled text
is as useless as misread text for search, translation and training.

Having the data is not the same as being able to use it. So what does it take?

## Part 2 — The work

Each step below is a decision someone makes, code someone runs, and work someone
is paid to do.

### Step 1 — Remove the noise
To the machine, the stains, the paper texture and the scanner noise all look like ink.
"""))

C.append(code("""\
denoised = cv2.fastNlMeansDenoising(scan, h=15)

zoom = (slice(230, 470), slice(80, 700))   # a few lines from column 1, enlarged
show(scan[zoom], denoised[zoom], titles=["Before: noise and stains", "After: denoised"], height=4)
"""))

C.append(md("""\
### Step 2 — Separate ink from paper
This step is called *binarization*: every pixel becomes pure black or pure white.
It also removes the text showing through from the back page.

To decide where black ends and white begins, we use *Otsu's method*: it reads the
brightness histogram of the page and automatically picks the cutoff that best
separates dark ink from light paper.
"""))

C.append(code("""\
_, bw = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

show(denoised[zoom], bw[zoom], titles=["Before: 256 shades of grey", "After: black or white only"], height=4)
"""))

C.append(md("""\
### Step 3 — Straighten the page
The page went into the scanner at an angle. OCR reads in straight lines, so we
measure the tilt of the text and rotate the page back.
"""))

C.append(code("""\
ink = np.column_stack(np.where(bw < 128))            # coordinates of every ink pixel
angle = cv2.minAreaRect(ink[:, ::-1])[-1]
if angle > 45:
    angle -= 90

h, w = bw.shape
M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
straight = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)

print(f"Measured tilt: {angle:.1f}°")
show(bw, straight, titles=["Tilted", "Straightened"], height=6)
"""))

C.append(md("""\
### Read it again
Same page. Same OCR engine. Only the preparation changed.
"""))

C.append(code("""\
clean_text = pytesseract.image_to_string(straight, lang="eng")

print(clean_text[:400])
print("…")
print(f"\\nBefore preparation: {CER(truth, raw_text):>5} % of characters wrong")
print(f"After preparation : {CER(truth, clean_text):>5} % of characters wrong")
"""))

C.append(md("""\
From about 60% wrong to under 2% wrong — and the two columns are now read in the
correct order. The page can be searched, translated and used for training. That
improvement is the value the preparation added.

## Part 3 — The gap software cannot close

Look closely at the result. Hausa uses the letters **ƙ, ɗ and ɓ**, and they change
meaning: *ɗari* (hundred) is not *dari* (cold weather). Did these letters survive?
"""))

C.append(code("""\
hooked = sum(clean_text.count(ch) for ch in "ƙɗɓƘƊƁ")
print(f"Hooked letters (ƙ ɗ ɓ) in the true text  : {sum(truth.count(c) for c in 'ƙɗɓƘƊƁ')}")
print(f"Hooked letters in the OCR output         : {hooked}\\n")

for wanted, frag in [("ƙarfafa", "arfafa"), ("ɗari uku", "dari uku"), ("ɓangaren", "angaren")]:
    line = next((l for l in clean_text.splitlines() if frag in l.lower()), "")
    got = " ".join(w for w in line.split() if frag.split()[0] in w.lower() or w.lower() in frag.split())
    print(f'  true word "{wanted}" → OCR wrote "{got or "?"}"   in: {line.strip()}')
"""))

C.append(md("""\
All of them are gone, and no error message told us. The reason: we read a Hausa
newspaper with an *English* OCR model. And we had no other choice:
"""))

C.append(code("""\
langs = pytesseract.get_languages()
print("Language models available here:", langs)
print("\\nTesseract offers about 130 language models. Hausa ('hau') is not one of them.")
print("About 80 million people speak Hausa. No one has built the training data yet.")
"""))

C.append(md("""\
Where a community *has* built the data, the same free software keeps the language
intact. Yoruba has a Tesseract model. Compare the two models on a Yoruba sentence
with full tone marks:
"""))

C.append(code("""\
ysnippet = cv2.imread(os.path.join(DATA, "yoruba_snippet.png"), cv2.IMREAD_GRAYSCALE)
ytruth = open(os.path.join(DATA, "yoruba_ground_truth.txt")).read().strip()

print(f"true text    : {ytruth}")
print(f"English model: {' '.join(pytesseract.image_to_string(ysnippet, lang='eng').split())}")
print(f"Yoruba model : {' '.join(pytesseract.image_to_string(ysnippet, lang='yor').split())}")
"""))

C.append(md("""\
The English model drops the tone marks. The Yoruba model keeps most of them,
because people collected Yoruba training data and built a model with it.
**Models exist only for languages that have datasets.**

### Why this matters for the continent

When pages become text at scale:

- **Archives become searchable** — a journalist can find every mention of a town across 40 years of print.
- **The text becomes training data** — machine translation, spell checkers and LLMs for African languages are built from text like this.
- **Heritage is preserved** — paper burns and fades; text files can be copied.
- **Knowledge becomes discoverable** — data that no one can find might as well not exist.

One more thing. To measure our results today, we needed the *ground truth*: a
correct transcription made by a person. The cleaning steps can be automated. The
truth cannot. Transcription, annotation and review are human work, and that work
adds value to the data. Who owns that added value? That is the subject of the
next session, on IP, copyright and licensing.

---
*The appendix below is for self-study after the workshop.*
"""))

C.append(md("""\
---
# Appendix — for self-study
This part is not in the live demo.

## A1. Smarter binarization
Otsu's method worked here because the lighting on our page is fairly even. Photos
taken with a phone often have shadows. *Adaptive* thresholding decides
ink-or-paper for each small region of the page instead of once for the whole page.
"""))

C.append(code("""\
adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)
show(bw[zoom], adaptive[zoom], titles=["Global (Otsu)", "Adaptive (local)"], height=4)
"""))

C.append(md("""\
## A2. Layout analysis — finding the text
Before reading, an OCR engine splits the page into columns, blocks, lines and
words. Newspapers with several columns are where the reading order often goes
wrong. Tesseract can show the word boxes it found:
"""))

C.append(code("""\
boxes = pytesseract.image_to_data(straight, lang="eng", output_type=pytesseract.Output.DICT)
overlay = cv2.cvtColor(straight, cv2.COLOR_GRAY2BGR)
for i, conf in enumerate(boxes["conf"]):
    if int(conf) > 40:
        x, y, bw_, bh = boxes["left"][i], boxes["top"][i], boxes["width"][i], boxes["height"][i]
        cv2.rectangle(overlay, (x, y), (x + bw_, y + bh), (200, 30, 30), 2)
plt.figure(figsize=(14, 9))
plt.imshow(overlay)
plt.axis("off")
plt.title("Each box is one word the engine located (confidence > 40)")
plt.show()
"""))

C.append(md("""\
## A3. Neural OCR
Modern OCR systems (TrOCR, docTR, Surya, and the Google and Azure APIs) replace
this hand-tuned pipeline with trained models. They handle damaged pages and
handwriting much better — but they need even more training data, which most
African languages do not have yet. To try one on a Colab GPU:

```python
# !pip install python-doctr[torch]
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# result = ocr_predictor(pretrained=True)(DocumentFile.from_images([f"{DATA}/newspaper_raw.png"]))
```

## A4. How this demo was made
The "1972 newspaper" was created for this workshop and then artificially aged by
[`scripts/make_ocr_assets.py`](../scripts/make_ocr_assets.py). That is how we know
the exact ground truth and can show honest error rates. A real archive has no
ground truth until a person writes it. Tools like
[AfriAnnotate](https://label.afriannotate.org) support that work: the OCR engine
suggests text, people correct it, and the corrected text becomes training data
for better models.

## A5. Where to go next
- [Tesseract language models](https://github.com/tesseract-ocr/tessdata) — see which languages have a model and which do not
- [Masakhane](https://www.masakhane.io/) — a grassroots community for NLP in African languages
- [Lacuna Fund](https://lacunafund.org/) — funds the creation of African language datasets
- [British Library Endangered Archives Programme](https://eap.bl.uk/) — digitized African newspapers and manuscripts
"""))

save(nb1, "01_ocr_scan_to_text.ipynb")

# =====================================================================
# Notebook 2 — Audio
# =====================================================================

nb2 = nbf.v4.new_notebook()
C = nb2.cells

C.append(md("""\
# From Field Recording to Model-Ready Speech
### *Daga rikodi zuwa corpus* — Deep Learning Indaba, Lagos 2026

Radio stations, oral historians, places of worship and the phones in this room
hold thousands of hours of African speech. None of it is a dataset yet.

We have seen a scanned page gain value. Now we do the same with sound. One rough
field recording goes in. A small, clean, labelled speech corpus comes out.

The recording is real Hausa: ten words spoken by a Hausa speaker (from Wikimedia
Commons, CC0 license). We joined them into one tape the way a field recorder
would capture them: background noise, electrical hum, and a speaker who moves
closer to and further from the microphone.

Note: cells with a grey title contain setup code. Double-click them to read the
code inside.
"""))

C.append(code("""\
#@title Setup — installs and data (double-click to read the code) { display-mode: "form" }
# Installs the audio helpers, downloads the workshop data, and defines the
# listen and plot utilities.
import os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "noisereduce"], capture_output=True)

REPO_URL = "https://github.com/abumafrim/indaba2026-ocr-audio"  # update if you fork
if os.path.exists("../data"):
    ROOT = ".."
elif os.path.exists("data"):
    ROOT = "."
else:
    ROOT = "indaba2026-ocr-audio"
    if not os.path.exists(os.path.join(ROOT, "data")):   # not downloaded yet
        result = subprocess.run(["git", "clone", "-q", REPO_URL + ".git", ROOT],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Could not download the workshop data. Git said:\n"
                               + result.stderr.strip())
DATA = os.path.join(ROOT, "data")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import soundfile as sf
from IPython.display import Audio, display

def listen(y, sr, label=""):
    if label:
        print(label)
    display(Audio(y, rate=sr))

def waveform(*sigs, sr, titles=None, height=2.6):
    fig, axes = plt.subplots(len(sigs), 1, figsize=(14, height * len(sigs)), squeeze=False)
    for ax, y, t in zip(axes[:, 0], sigs, (titles or [""] * len(sigs))):
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title(t, fontsize=12)
        ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.show()

def spectrogram(*sigs, sr, titles=None):
    fig, axes = plt.subplots(1, len(sigs), figsize=(7.5 * len(sigs), 4.5), squeeze=False)
    for ax, y, t in zip(axes[0], sigs, (titles or [""] * len(sigs))):
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(t, fontsize=12)
    plt.tight_layout()
    plt.show()

word_list = open(os.path.join(DATA, "word_list.txt")).read().split()
print("Setup complete.  Expecting", len(word_list), "words:", ", ".join(word_list))
"""))

C.append(md("""\
## Part 1 — The asset

One tape from a word-list recording session. Recording word lists is a standard
first step when documenting a language. Let us open the file exactly as the
recorder saved it.
"""))

C.append(code("""\
tape_path = os.path.join(DATA, "field_recording_raw.wav")
y_raw, sr_raw = librosa.load(tape_path, sr=None, mono=False)

print(f"channels    : {y_raw.shape[0]}  (stereo)")
print(f"sample rate : {sr_raw:,} per second")
print(f"duration    : {y_raw.shape[1] / sr_raw:.1f} s")
print(f"file size   : {os.path.getsize(tape_path) / 1e6:.1f} MB")
listen(y_raw, sr_raw, "The raw tape. Listen for the hiss, the hum, and the uneven voice:")
"""))

C.append(code("""\
waveform(y_raw[0], sr=sr_raw, titles=["The raw tape: 10 words are inside — some loud, some barely above the noise"])
"""))

C.append(md("""\
To a model, this file is only numbers: 48,000 of them per second, per channel.
Nothing in the file says where the words start, what was said, or who spoke.

## Part 2 — The work

### Step 1 — One channel is enough
The "stereo" from a field recorder is really the same microphone twice. Speech
models expect one channel.
"""))

C.append(code("""\
y = librosa.to_mono(y_raw)
print(f"before: {y_raw.shape}  →  after: {y.shape}   (half the numbers, nothing lost)")
"""))

C.append(md("""\
### Step 2 — Down to 16,000 samples per second
Speech models usually work at 16 kHz. The higher rate mostly stores frequencies
that human speech does not use. Listen to both. Can you hear a difference?
"""))

C.append(code("""\
y16 = librosa.resample(y, orig_sr=sr_raw, target_sr=16_000)
sr = 16_000

listen(y, sr_raw, "At 48,000 samples per second:")
listen(y16, sr, "At 16,000 — one third of the data:")
"""))

C.append(md("""\
### Step 3 — Remove the room
The recorder captured the hum and the hiss together with the voice. We give the
algorithm one second of "silence" — which is really pure noise — and it subtracts
that noise from the whole tape.
"""))

C.append(code("""\
y_clean = nr.reduce_noise(y=y16, sr=sr, y_noise=y16[:sr], stationary=True)

listen(y16, sr, "Before:")
listen(y_clean, sr, "After noise reduction:")
"""))

C.append(md("""\
### What the model sees
Models do not hear sound. They look at it, as a *spectrogram*: time runs left to
right, pitch runs bottom to top. Compare the two pictures:
"""))

C.append(code("""\
spectrogram(y16, y_clean, sr=sr, titles=["Before: speech buried in noise", "After: ten clear word shapes"])
"""))

C.append(md("""\
### Step 4 — Cut the tape into clips
Now that the silence is truly silent, the machine can find the words by itself.
We also make every clip equally loud.
"""))

C.append(code("""\
intervals = librosa.effects.split(y_clean, top_db=30)

# join pieces that are closer than 0.3 s — they belong to the same word
segments = []
for s, e in intervals:
    if segments and s - segments[-1][1] < int(0.3 * sr):
        segments[-1][1] = e
    else:
        segments.append([s, e])

print(f"words on the tape: {len(word_list)}   segments found: {len(segments)}\\n")

clips = {}
pad = int(0.05 * sr)
for word, (s, e) in zip(word_list, segments):
    clip = y_clean[max(0, s - pad): e + pad]
    clips[word] = clip / (np.max(np.abs(clip)) + 1e-9) * 0.9   # equal loudness
    print(f"  {word:<10} {(e - s) / sr:4.1f} s")
"""))

C.append(md("""\
Ten words on the tape, ten clips found. Each clip now has a name. A sound with a
label is a training example.
"""))

C.append(code("""\
listen(clips["Bahaushe"], sr, "Bahaushe — a Hausa person:")
listen(clips["Baƙauye"], sr, "Baƙauye — a villager. Note the letter ƙ, the same letter the OCR lost:")
"""))

C.append(md("""\
### The result — a small corpus
A model builder never receives "a tape". They receive clips plus a *manifest*:
file name, text, duration, speaker, license. Let us finish the job:
"""))

C.append(code("""\
outdir = "hausa_words_corpus"
os.makedirs(outdir, exist_ok=True)

manifest = []
for i, (word, clip) in enumerate(clips.items()):
    fname = f"{i:02d}_{word}.wav"
    sf.write(os.path.join(outdir, fname), clip, sr, subtype="PCM_16")
    manifest.append((fname, word, f"{len(clip) / sr:.2f}", "Gwanki", "CC0"))

print(f"{'file':<18} {'text':<10} {'sec':>5}  speaker  license")
for row in manifest:
    print(f"{row[0]:<18} {row[1]:<10} {row[2]:>5}  {row[3]:<8} {row[4]}")

corpus_mb = sum(os.path.getsize(os.path.join(outdir, f)) for f in os.listdir(outdir)) / 1e6
print(f"\\nraw tape: 4.4 MB, unlabelled  →  corpus: {corpus_mb:.2f} MB, labelled and uniform")
"""))

C.append(md("""\
## Part 3 — From 23 seconds to 23,000 hours

Everything above was one tape and ten words. A usable speech corpus is thousands
of hours, so every step must be automated, checked, and paid for. Between a radio
archive and a Hausa speech model sit the people in this room:

- **Media houses** own the recordings — the raw material this pipeline starts from.
- **Speakers and annotators** add the transcripts — the ground truth a machine cannot create.
- **Engineers** scale up the cleaning you just saw.
- **Lawyers** answer the question this notebook has been raising quietly.

Look at the manifest again. *Speaker* and *license* are part of the data. Whose
voice is on the tape? Who agreed to what? Who owns the value that cleaning and
labelling added? That is where the next session, on IP, copyright and licensing,
begins.

> Data is not yet a dataset. The difference is human work, and that work has owners.

---
*The appendix below is for self-study after the workshop.*
"""))

C.append(md("""\
---
# Appendix — for self-study
This part is not in the live demo.

## A1. Can speech recognition handle Hausa?
The OCR notebook found no Hausa model in Tesseract. Speech is only a little
better. OpenAI's Whisper lists Hausa, but it was trained on very little Hausa
audio. Try it and judge for yourself:

```python
# !pip install -q faster-whisper
# from faster_whisper import WhisperModel
# model = WhisperModel("small")
# segs, info = model.transcribe("hausa_words_corpus/00_Bahaushe.wav", language="ha")
# print(list(segs))
```

Speech recognition for low-resource languages improves only when corpora like the
one you just built exist at much larger scale — see
[Common Voice Hausa](https://commonvoice.mozilla.org/ha),
[NaijaVoices](https://naijavoices.com/) (about 1,800 hours of Hausa, Igbo and
Yoruba), [Africa Next Voices](https://africanvoices.io/dataset), and
[BibleTTS](https://masakhane-io.github.io/bibleTTS/) (high-quality Hausa audio
for speech synthesis).

## A2. Better voice activity detection
Our `librosa.effects.split` step uses loudness only. That works in a quiet room
but fails in a market or next to a radio. Production pipelines use a trained
voice-activity-detection model:

```python
# import torch
# vad, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
# speech_ts = utils[0](torch.from_numpy(y_clean), vad, sampling_rate=16_000)
```

## A3. Quality control at corpus scale
With thousands of clips, you check the data with statistics, not with your ears:
clip durations, loudness, clipping, wrong sample rates, near-duplicates. Charts
like these are what speech-data teams look at every day:
"""))

C.append(code("""\
durs = [len(c) / sr for c in clips.values()]
peaks = [float(np.max(np.abs(c))) for c in clips.values()]

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
axes[0].hist(durs, bins=8)
axes[0].set_title("Clip durations (s) — outliers usually mean a split error")
axes[1].hist(peaks, bins=8)
axes[1].set_title("Peak levels — after normalization, all near 0.9")
plt.tight_layout()
plt.show()
"""))

C.append(md("""\
## A4. How this demo was made
The tape is 10 real CC0 Hausa recordings by Wikimedia contributor **Gwanki**
([credits](../data/source_words/CREDITS.txt)), joined and degraded by
[`scripts/make_audio_assets.py`](../scripts/make_audio_assets.py) to imitate
field conditions. Every problem you watched us fix was added on purpose, so the
word-level ground truth was known. Real archives have no ground truth until
people make it. Transcription and review platforms such as
[AfriAnnotate](https://label.afriannotate.org) exist for that work — and they keep
track of consent and credit, so that information survives all the way to the model.

## A5. Where to go next
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) — contribute your voice in your language
- [Lacuna Fund](https://lacunafund.org/) — funding for African speech datasets
- [Masakhane](https://www.masakhane.io/) — African NLP community, speech included
- `librosa`, `noisereduce`, `silero-vad`, `faster-whisper` — the exact tools used today
"""))

save(nb2, "02_audio_tape_to_corpus.ipynb")
