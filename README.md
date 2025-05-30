# Kontext UI - Open Source Fal Kontext Demo

## Overview

Kontext UI is an open source interface for the [Fal Kontext API](https://fal.ai/models/fal-ai/flux-pro/kontext/api?platform=python), allowing you to edit images using natural language prompts. The system supports robust job tracking, logging, and automatic resumption of incomplete jobs.

## Features
- Edit images using the Fal Kontext API with a simple UI
- Upload local images (via ImgBB) or use image URLs
- Tracks all in-progress jobs in a `.active` file (never pushed to git)
- Resumes and saves results for any jobs left in progress if the app restarts
- Logs all activity (prompt, input URL, output URL) to a rolling `activity.log` (with backup)
- Ensures all output images are saved locally if the 'Save output image' box is checked

## Installation

### 1. Clone the repository
```sh
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

## Environment Variables

You must set the following environment variables before running the app:

- `FAL_KEY` — Your Fal API key (see https://fal.ai/models/fal-ai/flux-pro/kontext/api?platform=python)
- `IMGBB_API_KEY` — Your ImgBB API key (for uploading local images)

Example (Linux/macOS):
```sh
export FAL_KEY="your_fal_api_key"
export IMGBB_API_KEY="your_imgbb_api_key"
```

Example (Windows, PowerShell):
```powershell
$env:FAL_KEY="your_fal_api_key"
$env:IMGBB_API_KEY="your_imgbb_api_key"
```

## Usage

Run the app with:
```sh
python kontext-ui.py --verbose
```

- The UI will be available at the address shown in the terminal (default: http://localhost:7500)
- All in-progress jobs are tracked in `.active` and will be resumed on startup
- All activity is logged to `activity.log` (rolls over to `old_activity.log` at 10MB)
- Output images are saved to the `output/YYYYMMDD/` directory if 'Save output image' is checked

## Notes
- `.active`, `activity.log`, and `old_activity.log` are automatically added to `.gitignore`
- If the app is restarted, any jobs in progress will be resumed and their results saved
- If you encounter issues with image saving or job tracking, check the debug output (run with `--verbose`)

## License
MIT