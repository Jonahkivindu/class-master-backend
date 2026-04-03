from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os
import uvicorn
import time

app = FastAPI()

# Load model once
model = WhisperModel("base", device="cpu", compute_type="int8")

# Global stopwatch state
stopwatch_running = False
stopwatch_start = None
stopwatch_elapsed = 0.0


@app.post("/stopwatch/start")
def start_stopwatch():
    global stopwatch_running, stopwatch_start
    stopwatch_running = True
    stopwatch_start = time.perf_counter()
    return {"message": "Stopwatch started"}


@app.post("/stopwatch/stop")
def stop_stopwatch():
    global stopwatch_running, stopwatch_start, stopwatch_elapsed
    if stopwatch_running:
        stopwatch_elapsed += time.perf_counter() - stopwatch_start
        stopwatch_running = False
    return {"message": "Stopwatch stopped", "elapsed_sec": round(stopwatch_elapsed, 2)}


@app.get("/stopwatch/status")
def stopwatch_status():
    global stopwatch_running, stopwatch_start, stopwatch_elapsed
    if stopwatch_running:
        current_elapsed = stopwatch_elapsed + (time.perf_counter() - stopwatch_start)
    else:
        current_elapsed = stopwatch_elapsed
    return {
        "running": stopwatch_running,
        "elapsed_sec": round(current_elapsed, 2)
    }


@app.post("/whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save file
    temp_name = f"temp_{file.filename}"
    with open(temp_name, "wb") as f:
        f.write(await file.read())

    # 🤖 TRANSCRIBE
    transcribe_start = time.perf_counter()
    segments, _ = model.transcribe(temp_name)
    text = " ".join([s.text for s in segments])
    transcribe_time = time.perf_counter() - transcribe_start

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

    # Clean up
    os.remove(temp_name)

    # Stopwatch status at the end
    if stopwatch_running:
        current_elapsed = stopwatch_elapsed + (time.perf_counter() - stopwatch_start)
    else:
        current_elapsed = stopwatch_elapsed

    return {
        "transcript": text,
        "summary": summary,
        "timing": {
            "transcription_time_sec": round(transcribe_time, 2),
            "stopwatch_elapsed_sec": round(current_elapsed, 2),
            "stopwatch_running": stopwatch_running
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
