# audio.py  ← YOUR FILE (Phase 3)
from faster_whisper import WhisperModel
import re
import soundfile as sf
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8_float16" if device == "cuda" else "int8"
print(f"Audio: {device} + {compute_type}")

model = WhisperModel("tiny", device=device, compute_type=compute_type)

FILLERS = ["um", "uh", "ah", "like"]

def audio_phase_score(audio_path):
    segments, _ = model.transcribe(audio_path)
    text = " ".join([s.text for s in segments])
    
    data, sr = sf.read(audio_path)
    duration = len(data) / sr
    
    clean = re.sub(r"[^a-z\s]", " ", text.lower())
    words = len(clean.split())
    wpm = words / (duration / 60)
    
    fillers = sum(1 for f in FILLERS if f in text.lower())
    fluency = round(1.0 - min(fillers/5, 0.5) + (wpm-100)/200, 2)
    
    return {
        "transcript": text,
        "wpm": round(wpm, 1),
        "fillers": fillers,
        "fluency_score": max(0, min(1.0, fluency))
    }

if __name__ == "__main__":
    print("✅ Audio Phase 3 LOADED!")
