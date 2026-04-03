from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os
import uvicorn
import time

app = FastAPI()

# Load model once
model = WhisperModel("base", device="cpu", compute_type="int8")


def timed_step(label: str, func, *args, **kwargs):
    """Helper to measure execution time of a function call."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[⏱️] {label} took {elapsed:.2f} sec")
    return result, elapsed


@app.post("/whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    # ⏱️ START TIMER
    start_time = time.perf_counter()

    # Save file
    save_start = time.perf_counter()
    temp_name = f"temp_{file.filename}"
    with open(temp_name, "wb") as f:
        f.write(await file.read())
    upload_time = time.perf_counter() - save_start
    print(f"[⏱️] File received in {upload_time:.2f} sec")

    # 🤖 TRANSCRIBE
    (segments, _), transcribe_time = timed_step("Transcription", model.transcribe, temp_name)
    text = " ".join([s.text for s in segments])

    # 🧠 ENTITY EXTRACTION
    def extract_entities(text: str):
        words = text.split()
        entities = [
            w.strip(".,!?:")
            for w in words
            if len(w) > 3 and w[0].isupper()
        ]
        return list(set(entities))

    unique_names, entity_time = timed_step("Entity Extraction", extract_entities, text)

    # 📝 SUMMARY
    def build_summary(names, text):
        summary = "### 💡 KEY NAMES & TOPICS:\n"
        summary += ", ".join(names[:15])
        summary += "\n\n### 📝 QUICK NOTES:\n"
        summary += text[:400] + "..."
        return summary

    summary, summary_time = timed_step("Summary Generation", build_summary, unique_names, text)

    # ⏱️ TOTAL TIME
    total_time = time.perf_counter() - start_time
    print(f"[✅] Total processing time: {total_time:.2f} sec")

    # Clean up
    os.remove(temp_name)

    return {
        "transcript": text,
        "summary": summary,
        "timing": {
            "upload_time_sec": round(upload_time, 2),
            "transcription_time_sec": round(transcribe_time, 2),
            "entity_extraction_time_sec": round(entity_time, 2),
            "summary_time_sec": round(summary_time, 2),
            "total_time_sec": round(total_time, 2)
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
