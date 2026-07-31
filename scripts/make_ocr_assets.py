"""Generate the OCR demo assets.

Creates a synthetic "aged newspaper" page in Hausa. Because we render the
page ourselves, we know the exact ground-truth text — which lets the demo
report honest character error rates. With a real archive scan you don't have
that luxury, which is itself one of the workshop's talking points.

Outputs (written to ../data/):
  newspaper_raw.png     — the degraded scan (skew, stains, noise, bleed-through)
  newspaper_clean.png   — the pristine render, for reference
  newspaper_ground_truth.txt
  yoruba_snippet.png    — clean Yoruba sentence with full diacritics
  yoruba_ground_truth.txt
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

random.seed(14)
np.random.seed(14)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Fonts: macOS supplemental fonts locally, DejaVu on Linux/Colab.
FONT_CANDIDATES = {
    "serif": [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ],
    "serif_bold": [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ],
    "unicode": [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}


def load_font(kind, size):
    for path in FONT_CANDIDATES[kind]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


MASTHEAD = "LABARAN YAU"
DATELINE = "Litinin, 14 ga Agusta, 1972  —  Farashi: Kobo Hamsin"

COLUMNS = [
    (
        "Manoma a jihar Kano sun sami babban amfanin gona na masara a bana. "
        "Gwamnati ta yi alkawarin sayen dukkan amfanin gona a farashi mai "
        "kyau don ƙarfafa gwiwar manoma. Shugabannin ƙauyuka sun ce ruwan "
        "sama na bana ya yi yawa kuma an raba iri masu inganci a kan lokaci. "
        "An buɗe sabuwar makarantar firamare a ƙauyen Dawakin Kudu. An yi "
        "rajistar yara fiye da ɗari uku a aji na farko."
    ),
    (
        "Malamai sun ce suna buƙatar ƙarin littattafai da sababbin kujeru "
        "don koyar da ɗalibai yadda ya kamata. An gina sabuwar kasuwar kifi "
        "a ɓangaren gabashin birnin Legas kusa da tashar jiragen ruwa. "
        "Masunta yanzu suna iya sayar da kifinsu a wuri mai tsabta. Jama'a "
        "da yawa sun yi farin ciki saboda farashin kifi ya sauka a kasuwa "
        "kuma jigilar amfanin gona ta zama mai sauƙi fiye da shekarun baya."
    ),
]

YORUBA_TEXT = "Ọjọ́ dára púpọ̀. Àwọn ọmọdé ń kọrin ní ilé ìwé. Olùkọ́ fẹ́ràn iṣẹ́ rẹ̀."


def wrap_text(draw, text, font, max_width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def render_newspaper():
    W, H = 1400, 1000
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)

    masthead_font = load_font("serif_bold", 68)
    dateline_font = load_font("serif", 34)
    body_font = load_font("serif", 30)

    w = draw.textlength(MASTHEAD, font=masthead_font)
    draw.text(((W - w) / 2, 40), MASTHEAD, font=masthead_font, fill=0)
    draw.line([(80, 130), (W - 80, 130)], fill=0, width=3)
    w = draw.textlength(DATELINE, font=dateline_font)
    draw.text(((W - w) / 2, 142), DATELINE, font=dateline_font, fill=0)
    draw.line([(80, 182), (W - 80, 182)], fill=0, width=2)

    col_width, margin, gutter, y0 = 590, 80, 60, 220
    for i, col_text in enumerate(COLUMNS):
        x = margin + i * (col_width + gutter)
        y = y0
        for line in wrap_text(draw, col_text, body_font, col_width):
            draw.text((x, y), line, font=body_font, fill=0)
            y += 42
    draw.line([(W / 2, 210), (W / 2, H - 60)], fill=0, width=1)
    return img


def degrade(img):
    """Apply the classic ailments of an archive scan."""
    W, H = img.size

    # Aged paper: sepia background with blotchy tone
    paper = np.full((H, W), 205, dtype=np.float64)
    paper += np.random.normal(0, 6, (H, W))
    page = np.array(img, dtype=np.float64)
    aged = np.minimum(page, paper)

    # Ink bleed-through from the reverse side: faint mirrored copy of the page
    ghost = np.array(ImageOps.mirror(img), dtype=np.float64)
    aged = aged - 0.16 * (255 - ghost)

    aged = np.clip(aged, 0, 255).astype(np.uint8)
    out = Image.fromarray(aged, "L")

    # Stains and foxing spots
    stain = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(stain)
    for _ in range(14):
        cx, cy = random.randint(0, W), random.randint(0, H)
        r = random.randint(30, 130)
        sd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=random.randint(25, 60))
    stain = stain.filter(ImageFilter.GaussianBlur(25))
    out = Image.fromarray(
        np.clip(np.array(out, dtype=np.int16) - np.array(stain, dtype=np.int16) // 2, 0, 255).astype(np.uint8)
    )

    # Crooked placement on the scanner bed (page margins absorb the shift,
    # and the near-paper fill keeps the canvas free of dark corner wedges)
    out = out.rotate(-2.1, expand=False, fillcolor=200, resample=Image.BICUBIC)

    # Sensor noise and loss of sharpness
    arr = np.array(out, dtype=np.float64) + np.random.normal(0, 14, out.size[::-1])
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    out = out.filter(ImageFilter.GaussianBlur(0.6))
    out = ImageEnhance.Contrast(out).enhance(0.82)
    return out


def render_yoruba():
    font = load_font("unicode", 40)
    img = Image.new("L", (1200, 110), 245)
    draw = ImageDraw.Draw(img)
    draw.text((30, 28), YORUBA_TEXT, font=font, fill=10)
    return img


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    clean = render_newspaper()
    clean.save(os.path.join(DATA_DIR, "newspaper_clean.png"))
    degrade(clean).save(os.path.join(DATA_DIR, "newspaper_raw.png"))

    ground_truth = MASTHEAD + "\n" + DATELINE + "\n" + "\n".join(COLUMNS)
    with open(os.path.join(DATA_DIR, "newspaper_ground_truth.txt"), "w") as f:
        f.write(ground_truth + "\n")

    render_yoruba().save(os.path.join(DATA_DIR, "yoruba_snippet.png"))
    with open(os.path.join(DATA_DIR, "yoruba_ground_truth.txt"), "w") as f:
        f.write(YORUBA_TEXT + "\n")
    print("assets written to", os.path.abspath(DATA_DIR))


if __name__ == "__main__":
    main()
