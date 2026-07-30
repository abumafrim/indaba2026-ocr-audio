"""Generate the audio demo asset: a simulated Hausa word-list field recording.

Takes 10 real Hausa word recordings (CC0, Wikimedia Commons "Hausa
pronunciation" category, speaker: Gwanki — see data/source_words/CREDITS.txt)
and splices them into one continuous "elicitation session" tape the way a
field recorder would capture it: 48 kHz stereo, background noise and hum,
uneven loudness between words, silence between items.

The notebook then reverses each ailment step by step and ends with the tape
split back into 10 clean, uniform, labelled clips — a miniature speech corpus.

Outputs (data/):
  field_recording_raw.wav   — 48 kHz stereo, noisy, ~35 s
  word_list.txt             — the 10 words in spoken order (ground truth)
"""
import glob
import os

import librosa
import numpy as np
import soundfile as sf

rng = np.random.default_rng(7)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SR = 48_000

WORD_ORDER = [
    "Bahaushe", "Baƙauye", "Aljihu", "Asabar",
    "Awa_(lokaci)", "Awara", "Azanci", "Bayani", "Bayyana", "Bazara",
]


def load_word(name):
    path = os.path.join(BASE, "source_words", f"{name}.wav")
    if not os.path.exists(path):
        return None
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    return y


def main():
    words, order = [], []
    for name in WORD_ORDER:
        y = load_word(name)
        if y is None:
            continue
        words.append(y)
        order.append(name.replace("_(lokaci)", ""))

    # Splice with silences and uneven per-word gain (mic distance varies)
    pieces = [np.zeros(int(SR * 1.2))]
    for y in words:
        gain = rng.uniform(0.25, 1.0)
        pieces.append(y * gain)
        pieces.append(np.zeros(int(SR * rng.uniform(0.9, 1.8))))
    tape = np.concatenate(pieces)

    # Field ailments: broadband hiss + 50 Hz mains hum + low rumble
    t = np.arange(len(tape)) / SR
    hiss = rng.normal(0, 0.012, len(tape))
    hum = 0.02 * np.sin(2 * np.pi * 50 * t) + 0.01 * np.sin(2 * np.pi * 100 * t)
    rumble = librosa.resample(rng.normal(0, 1.0, len(tape) // 400 + 200), orig_sr=SR // 400, target_sr=SR)
    rumble = np.resize(rumble, len(tape)) * 0.015
    noisy = tape + hiss + hum + rumble

    # Fake stereo the way cheap recorders do: same mic, slightly different noise
    right = noisy + rng.normal(0, 0.004, len(noisy))
    stereo = np.stack([noisy, right], axis=1)
    stereo = np.clip(stereo, -1, 1)

    sf.write(os.path.join(BASE, "field_recording_raw.wav"), stereo, SR, subtype="PCM_16")
    with open(os.path.join(BASE, "word_list.txt"), "w") as f:
        f.write("\n".join(order) + "\n")

    dur = len(tape) / SR
    mb = os.path.getsize(os.path.join(BASE, "field_recording_raw.wav")) / 1e6
    print(f"tape: {dur:.1f}s, {len(order)} words, {mb:.1f} MB @ 48kHz stereo")


if __name__ == "__main__":
    main()
