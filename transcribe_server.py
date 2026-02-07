"""
Whisper Bulk Transcriber — Local web UI for batch transcription.
Uses faster-whisper with GPU acceleration.
"""

import os
import sys
import glob
import json
import time
import threading
import webbrowser
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# Optional: OpenAI API for emergency fast transcription
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

app = Flask(__name__, static_folder="static")

# Optional: stable-ts for accurate timestamps (simpler than stable-ts)
try:
    import stable_whisper
    HAS_STABLE_TS = True
except ImportError:
    HAS_STABLE_TS = False

# --- State ---
state = {
    "model_loaded": False,
    "model_name": "tiny",
    "device": "cuda",
    "engine": "faster-whisper",  # "faster-whisper" or "stable-ts"
    "jobs": [],  # List of {id, input_folder, output_folder, status, files, current_file, progress, errors}
    "is_running": False,
    "worker_status": "",
    "openai_available": HAS_OPENAI,
    "openai_key": os.environ.get("OPENAI_API_KEY", ""),
    "stable_ts_available": HAS_STABLE_TS,
}
model = None
stable_ts_model = None
worker_thread = None


def get_audio_files(folder):
    """Find all audio/video files in a folder."""
    extensions = ["*.mp3", "*.mp4", "*.m4a", "*.webm", "*.wav", "*.ogg", "*.flac", "*.aac", "*.wma"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(files)


def load_model():
    """Load the whisper model."""
    global model
    from faster_whisper import WhisperModel
    
    device = state["device"]
    compute = "float16" if device == "cuda" else "int8"
    
    try:
        model = WhisperModel(state["model_name"], device=device, compute_type=compute)
        state["model_loaded"] = True
        print(f"Model '{state['model_name']}' loaded on {device} ({compute})")
    except Exception as e:
        if device == "cuda":
            # Try CUDA with int8 before giving up to CPU
            print(f"CUDA float16 failed ({e}), trying CUDA with int8...")
            try:
                model = WhisperModel(state["model_name"], device="cuda", compute_type="int8")
                state["model_loaded"] = True
                state["device"] = "cuda"
                print(f"Model '{state['model_name']}' loaded on CUDA (int8)")
            except Exception as e2:
                print(f"CUDA int8 also failed ({e2}), falling back to CPU...")
                state["device"] = "cpu"
                model = WhisperModel(state["model_name"], device="cpu", compute_type="int8")
                state["model_loaded"] = True
                print(f"Model '{state['model_name']}' loaded on CPU")
        else:
            raise


def transcribe_file(filepath):
    """Transcribe a single file, return (text_with_timestamps, segments_list)."""
    segments, info = model.transcribe(filepath, language="en", beam_size=5, vad_filter=True)
    
    lines = []
    segments_data = []
    
    for seg in segments:
        # Format timestamp as MM:SS.ss
        start_fmt = f"{int(seg.start//60):02d}:{seg.start%60:05.2f}"
        end_fmt = f"{int(seg.end//60):02d}:{seg.end%60:05.2f}"
        text = seg.text.strip()
        
        lines.append(f"[{start_fmt} --> {end_fmt}] {text}")
        segments_data.append({
            "start_ms": int(seg.start * 1000),
            "end_ms": int(seg.end * 1000),
            "text": text
        })
    
    return "\n".join(lines), segments_data


def transcribe_file_stable_ts(filepath):
    """
    Hybrid approach: faster-whisper for speed + stable-ts for accurate alignment.
    Best of both worlds.
    """
    if not HAS_STABLE_TS:
        raise RuntimeError("stable-ts not installed. Run: pip install stable-ts")
    
    # Step 1: Fast transcription with faster-whisper
    segments_fw, info = model.transcribe(filepath, language="en", beam_size=5, vad_filter=True)
    
    # Collect full transcript text
    full_text = ""
    for seg in segments_fw:
        full_text += seg.text
    
    # Step 2: Load a stable-ts model and align the text with audio
    # This gives us accurate word-level timestamps
    model_st = stable_whisper.load_model(state["model_name"], device=state["device"])
    result = model_st.align(filepath, full_text, language="en")
    
    lines = []
    segments_data = []
    
    for seg in result.segments:
        start = seg.start
        end = seg.end
        text = seg.text.strip()
        
        # Format timestamp
        start_fmt = f"{int(start//60):02d}:{start%60:05.2f}"
        end_fmt = f"{int(end//60):02d}:{end%60:05.2f}"
        
        lines.append(f"[{start_fmt} --> {end_fmt}] {text}")
        
        # Include word-level timestamps
        words_data = []
        if hasattr(seg, 'words') and seg.words:
            for w in seg.words:
                words_data.append({
                    "word": w.word,
                    "start_ms": int(w.start * 1000),
                    "end_ms": int(w.end * 1000),
                })
        
        segments_data.append({
            "start_ms": int(start * 1000),
            "end_ms": int(end * 1000),
            "text": text,
            "words": words_data
        })
    
    return "\n".join(lines), segments_data


def worker():
    """Background worker that processes the job queue."""
    global model
    
    state["is_running"] = True
    state["worker_status"] = "Loading model..."
    
    # Only load local model if there are non-API jobs
    needs_local = any(j.get("mode") != "api" and j["status"] == "queued" for j in state["jobs"])
    if needs_local and not state["model_loaded"]:
        state["worker_status"] = f"Loading {state['model_name']} on {state['device']}... (this may take 30-60s)"
        load_model()
        state["worker_status"] = f"Model ready on {state['device']}"
    
    for job in state["jobs"]:
        if job["status"] == "cancelled":
            continue
        if job["status"] != "queued":
            continue
        
        job["status"] = "running"
        input_folder = job["input_folder"]
        output_folder = job["output_folder"]
        
        os.makedirs(output_folder, exist_ok=True)
        
        audio_files = get_audio_files(input_folder)
        job["files"] = [os.path.basename(f) for f in audio_files]
        job["total"] = len(audio_files)
        job["completed"] = 0
        job["errors"] = []
        
        if not audio_files:
            job["status"] = "done"
            job["errors"].append("No audio files found in folder")
            continue
        
        for i, filepath in enumerate(audio_files):
            if job["status"] == "cancelled":
                break
            
            basename = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(output_folder, f"{basename}.txt")
            
            # Skip already transcribed
            if os.path.exists(out_path):
                job["completed"] = i + 1
                job["current_file"] = f"Skipped (exists): {basename}"
                continue
            
            job["current_file"] = basename
            job["progress"] = i / len(audio_files) * 100
            
            try:
                start = time.time()
                if job.get("mode") == "api":
                    text = transcribe_file_api(filepath, job["api_key"])
                    segments_data = []  # API mode doesn't return segments
                elif state["engine"] == "stable-ts":
                    text, segments_data = transcribe_file_stable_ts(filepath)
                else:
                    text, segments_data = transcribe_file(filepath)
                elapsed = time.time() - start
                
                # Save text with timestamps
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Save JSON with segment data (for Echo ingestion)
                if segments_data:
                    json_path = os.path.join(output_folder, f"{basename}.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "file": os.path.basename(filepath),
                            "segments": segments_data
                        }, f, indent=2)
                
                job["completed"] = i + 1
                job["current_file"] = f"Done: {basename} ({elapsed:.1f}s, {len(segments_data)} segments)"
            except Exception as e:
                job["errors"].append(f"{basename}: {str(e)}")
                job["completed"] = i + 1
        
        if job["status"] != "cancelled":
            job["status"] = "done"
            job["progress"] = 100
            job["current_file"] = "Complete"
    
    state["is_running"] = False


# --- Routes ---

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "model_loaded": state["model_loaded"],
        "model_name": state["model_name"],
        "device": state["device"],
        "engine": state["engine"],
        "stable-ts_available": state["stable-ts_available"],
        "is_running": state["is_running"],
        "worker_status": state.get("worker_status", ""),
        "jobs": state["jobs"],
    })


@app.route("/api/browse", methods=["POST"])
def api_browse():
    """List folders/files for the folder browser."""
    data = request.json or {}
    path = data.get("path", "")
    
    if not path:
        # Return drives on Windows, root on Unix
        if sys.platform == "win32":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({"name": drive, "path": drive, "type": "drive"})
            return jsonify({"items": drives, "current": ""})
        else:
            path = "/"
    
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return jsonify({"error": "Not a directory"}), 400
    
    items = []
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                items.append({"name": name, "path": full, "type": "folder"})
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403
    
    parent = os.path.dirname(path) if path != os.path.sep else None
    if sys.platform == "win32" and len(path) <= 3:
        parent = ""  # Back to drive list
    
    # Count audio files in this folder
    audio_count = len(get_audio_files(path))
    
    return jsonify({
        "items": items,
        "current": path,
        "parent": parent,
        "audio_count": audio_count,
    })


@app.route("/api/add_folder", methods=["POST"])
def api_add_folder():
    """Add a folder to the transcription queue."""
    data = request.json
    input_folder = data.get("input_folder", "").strip()
    output_mode = data.get("output_mode", "subfolder")  # "subfolder" or "custom"
    custom_output = data.get("custom_output", "").strip()
    
    if not input_folder or not os.path.isdir(input_folder):
        return jsonify({"error": "Invalid input folder"}), 400
    
    if output_mode == "custom" and custom_output:
        output_folder = custom_output
    else:
        output_folder = os.path.join(input_folder, "transcripts")
    
    audio_files = get_audio_files(input_folder)
    
    job = {
        "id": len(state["jobs"]),
        "input_folder": input_folder,
        "output_folder": output_folder,
        "folder_name": os.path.basename(input_folder),
        "status": "queued",
        "files": [os.path.basename(f) for f in audio_files],
        "total": len(audio_files),
        "completed": 0,
        "current_file": "",
        "progress": 0,
        "errors": [],
    }
    
    state["jobs"].append(job)
    return jsonify({"ok": True, "job": job})


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start processing the queue."""
    global worker_thread
    
    if state["is_running"]:
        return jsonify({"error": "Already running"}), 400
    
    queued = [j for j in state["jobs"] if j["status"] == "queued"]
    if not queued:
        return jsonify({"error": "No jobs in queue"}), 400
    
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    
    return jsonify({"ok": True, "queued_count": len(queued)})


@app.route("/api/cancel/<int:job_id>", methods=["POST"])
def api_cancel(job_id):
    """Cancel a job."""
    for job in state["jobs"]:
        if job["id"] == job_id:
            job["status"] = "cancelled"
            return jsonify({"ok": True})
    return jsonify({"error": "Job not found"}), 404


@app.route("/api/clear_done", methods=["POST"])
def api_clear_done():
    """Remove completed/cancelled jobs from the list."""
    state["jobs"] = [j for j in state["jobs"] if j["status"] in ("queued", "running")]
    return jsonify({"ok": True})


@app.route("/api/add_folder_api", methods=["POST"])
def api_add_folder_api():
    """Add a folder for OpenAI API transcription (emergency fast mode)."""
    if not HAS_OPENAI:
        return jsonify({"error": "openai package not installed. Run: pip install openai"}), 400
    
    data = request.json
    input_folder = data.get("input_folder", "").strip()
    api_key = data.get("api_key", "").strip() or state["openai_key"]
    
    if not api_key:
        return jsonify({"error": "No OpenAI API key. Set OPENAI_API_KEY env var or enter in UI."}), 400
    if not input_folder or not os.path.isdir(input_folder):
        return jsonify({"error": "Invalid input folder"}), 400
    
    output_folder = os.path.join(input_folder, "transcripts")
    audio_files = get_audio_files(input_folder)
    
    job = {
        "id": len(state["jobs"]),
        "input_folder": input_folder,
        "output_folder": output_folder,
        "folder_name": os.path.basename(input_folder) + " ⚡API",
        "status": "queued",
        "mode": "api",
        "api_key": api_key,
        "files": [os.path.basename(f) for f in audio_files],
        "total": len(audio_files),
        "completed": 0,
        "current_file": "",
        "progress": 0,
        "errors": [],
    }
    
    state["jobs"].append(job)
    return jsonify({"ok": True, "job": job})


def transcribe_file_api(filepath, api_key):
    """Transcribe using OpenAI Whisper API."""
    client = OpenAI(api_key=api_key)
    
    # OpenAI API has a 25MB limit — check file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > 25:
        # For large files, need to split — for now just warn
        raise Exception(f"File too large for API ({size_mb:.1f}MB > 25MB limit). Use local mode.")
    
    with open(filepath, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
            response_format="text"
        )
    return result


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Update model settings (only when not running)."""
    global stable_ts_model
    if state["is_running"]:
        return jsonify({"error": "Cannot change settings while running"}), 400
    
    data = request.json
    if "engine" in data:
        if data["engine"] == "stable-ts" and not HAS_STABLE_TS:
            return jsonify({"error": "stable-ts not installed. Run: pip install stable-ts"}), 400
        state["engine"] = data["engine"]
        state["model_loaded"] = False
        stable_ts_model = None
    if "model_name" in data:
        state["model_name"] = data["model_name"]
        state["model_loaded"] = False
        stable_ts_model = None
    if "device" in data:
        state["device"] = data["device"]
        state["model_loaded"] = False
        stable_ts_model = None
    
    return jsonify({"ok": True, "engine": state["engine"], "model_name": state["model_name"], "device": state["device"]})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5678
    print(f"\n  Whisper Bulk Transcriber")
    print(f"  http://localhost:{port}\n")
    webbrowser.open(f"http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
