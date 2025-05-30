# Kontext UI - Open Source Fal Kontext Demo

## Overview

Kontext UI is an open source interface for the [Fal Kontext API](https://fal.ai/models/fal-ai/flux-pro/kontext/api?platform=python), allowing you to edit images using natural language prompts. The system supports robust job tracking, logging, and automatic resumption of incomplete jobs.

![image](https://github.com/user-attachments/assets/5751424a-9202-4bf2-80a8-7b3f5eecbffa)


## Features
- Edit images using the Fal Kontext API with a simple UI
- Upload local images or use image URLs (images are sent directly as base64 Data URIs)
- Upload a video and extract the first frame to use as the input image for editing (supports mp4, mov, avi, webm)
- Tracks all in-progress jobs in a local `.active` file
- Resumes and saves results for any jobs left in progress if the app restarts
- Logs all activity (prompt, input URL, output URL) to a rolling `activity.log` (with backup)
- Ensures all output images are saved locally if the 'Save output image' box is checked
- Save output images to the `output/YYYYMMDD/` directory, with filenames based on the image hash from the API response.
- Optionally, save the input image alongside the output image by enabling the 'Save Input Image with output' setting. The input image will be saved in the same directory, with `_input` appended to the filename, and will be converted to match the output filetype (e.g., JPEG or PNG).

## Installation

### 1. Clone the repository
```sh
# Replace <your-repo-url> with the actual URL
git clone <your-repo-url>
cd fal-kontext-ui
```

### 2. Create and activate a Python virtual environment (recommended)
```sh
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```
- **Note:** `opencv-python` is required for video frame extraction and is included in `requirements.txt`.

## Environment Variables

You must set the following environment variable before running the app:
- `FAL_KEY` — Your Fal API key (see https://fal.ai/models/fal-ai/flux-pro/kontext/api?platform=python)

Example (Linux/macOS):
```sh
export FAL_KEY="your_fal_api_key"
```
Example (Windows, PowerShell):
```powershell
$env:FAL_KEY="your_fal_api_key"
```

## Usage

Run the app with:
```sh
python kontext-ui.py --verbose
```

### Host and Port Options
You can specify the network interface and port for the Gradio UI using `--host` and `--port`:
```sh
python kontext-ui.py --host 0.0.0.0 --port 7500
```
- `--host` (default: `0.0.0.0`) sets the network interface to listen on (e.g., `127.0.0.1` for localhost only).
- `--port` (default: `7500`) sets the port for the UI.
- The UI will be available at the address shown in the terminal (e.g., http://localhost:7500)
- All in-progress jobs are tracked in `.active` and will be resumed on startup
- All activity is logged to `activity.log` (rolls over to `old_activity.log` at 10MB)
- Output images are saved to the `output/YYYYMMDD/` directory if 'Save output image' is checked
- **To use the video-to-image feature:** Upload a video file, click "Extract First Frame from Video", and the first frame will be set as the input image for editing

## Notes
- `.active`, `activity.log`, `old_activity.log`, and `output/` are automatically added to `.gitignore`
- If the app is restarted, any jobs in progress will be resumed and their results saved
- If you encounter issues with image saving or job tracking, check the debug output (run with `--verbose`)

## License
MIT