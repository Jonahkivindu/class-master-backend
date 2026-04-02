from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os
import uvicorn

app = FastAPI()

# Load model once on startup to save time
# 'base' is small enough for free cloud tiers
model = WhisperModel("base", device="cpu", compute_type="int8")

@app.post("/whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the incoming file from your phone
    temp_name = f"temp_{file.filename}"
    with open(temp_name, "wb") as f:
        f.write(await file.read())
    
    # Run Whisper AI
    segments, _ = model.transcribe(temp_name)
    text = " ".join([s.text for s in segments])
    
    # Smart Extraction for Names (Jonah, Mark, Tatu)
    words = text.split()
    entities = [w.strip(".,!?:") for w in words if len(w) > 3 and w[0].isupper()]
    unique_names = list(set(entities))
    
    # Create the summary points
    summary = f"### 💡 KEY NAMES & TOPICS:\n" + ", ".join(unique_names[:15])
    summary += f"\n\n### 📝 QUICK NOTES:\n" + text[:400] + "..."
    
    os.remove(temp_name)
    return {"transcript": text, "summary": summary}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)