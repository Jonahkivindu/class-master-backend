from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os
import uvicorn
import time

app = FastAPI()

# Load model once
model = WhisperModel("base", device="cpu", compute_type="int8")


@app.post("/whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    
    # ⏱️ START TIMER
    start_time = time.time()

    # Save file
    temp_name = f"temp_{file.filename}"
    with open(temp_name, "wb") as f:
        f.write(await file.read())

    # ⏱️ RECORDING TIME (approx since upload started)
    upload_time = time.time() - start_time
    print(f"[⏱️] File received in {upload_time:.2f} sec")

    # 🤖 TRANSCRIBE
    transcribe_start = time.time()
    segments, _ = model.transcribe(temp_name)
    text = " ".join([s.text for s in segments])
    transcribe_time = time.time() - transcribe_start

    print(f"[🤖] Transcription took {transcribe_time:.2f} sec")

    # 🧠 ENTITY EXTRACTION
    words = text.split()
    entities = [
        w.strip(".,!?:") 
        for w in words 
        if len(w) > 3 and w[0].isupper()
    ]
    unique_names = list(set(entities))

    # 📝 SUMMARY
    summary = "### 💡 KEY NAMES & TOPICS:\n"
    summary += ", ".join(unique_names[:15])

    summary += "\n\n### 📝 QUICK NOTES:\n"
    summary += text[:400] + "..."

    # ⏱️ TOTAL TIME
    total_time = time.time() - start_time

    print(f"[✅] Total processing time: {total_time:.2f} sec")

    # Clean up
    os.remove(temp_name)

    return {
        "transcript": text,
        "summary": summary,
        "timing": {
            "upload_time_sec": round(upload_time, 2),
            "transcription_time_sec": round(transcribe_time, 2),
            "total_time_sec": round(total_time, 2)
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
