# Whisper Bulk Transcriber

Local web UI for batch transcription of audio/video files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with GPU acceleration.

## Quick Start (Windows)

1. **Prerequisites:** Python 3.10+, NVIDIA GPU with CUDA
2. Double-click `start_transcriber.bat`
3. Browser opens automatically at `http://localhost:5678`

## Usage

1. **Browse** to your audio folders using the folder browser
2. **Add folders** individually or use "Add All Subfolders" for bulk
3. **Start Transcription** — transcripts save to a `transcripts/` subfolder in each input folder
4. Pick up the transcript `.txt` files from the output folders

## Features

- 📁 Folder browser with audio file detection
- 📚 Bulk mode: add all subfolders at once
- ⏭️ Skips already-transcribed files (safe to re-run)
- 📊 Real-time progress per folder
- ⚙️ Model selection (tiny/base/small/medium/large-v3)
- 🎮 GPU (CUDA) or CPU mode
- 🌐 Auto-launches browser

## Models

| Model | VRAM | Speed (30 min lecture) | Accuracy |
|-------|------|----------------------|----------|
| tiny | ~1 GB | ~1-2 min | OK |
| base | ~1 GB | ~2-3 min | Good |
| **small** | ~2 GB | **~3-5 min** | **Very Good** |
| medium | ~5 GB | ~8-12 min | Excellent |
| large-v3 | ~10 GB | ~15-20 min | Best |

**Recommended: `small`** for clear lecture audio on a GTX 1050 Ti (4GB VRAM).

## Supported Formats

mp3, mp4, m4a, webm, wav, ogg, flac, aac, wma

## Manual Install

```bash
pip install faster-whisper flask
python transcribe_server.py
```
