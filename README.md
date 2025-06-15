# Kontext UI - Open Source Fal Kontext Demo

## Overview

Kontext UI is an open source interface for the [Fal Kontext API](https://fal.ai/models/fal-ai/flux-pro/kontext/api?platform=python), allowing you to edit images using natural language prompts. The system supports robust job tracking, logging, and automatic resumption of incomplete jobs.

**SINGLE IMAGE MODE**
![image](https://github.com/user-attachments/assets/e1d0555f-8abf-4bb3-9a6e-bafc308d78db)


**MULTI-IMAGE MODE**
![image](https://github.com/user-attachments/assets/47e8ad9e-2e05-4693-8607-5c57a3c73924)



## Features
- Edit images using the Fal Kontext API with a simple UI
- **NEW: Multiple Input Mode!** Upload up to 4 images and use them together in a single prompt via the Kontext Max API (see below)
- Upload local images or use image URLs (images are sent directly as base64 Data URIs for single input, or uploaded to Fal storage for multi-input)
- Upload a video and extract the first frame to use as the input image for editing (supports mp4, mov, avi, webm)
- **Transform input images** with rotation and flipping options before sending to the API:
  - Rotate Input Image Left (90°) / Right (90°)
  - Flip Input Image Horizontal / Vertical
  - Apply transformations repeatedly (e.g., rotate left 4 times returns to original)
- Logs all activity (prompt, input URL, output URL) to a rolling `activity.log` (with backup)
- Ensures all output images are saved locally if the 'Save output image' box is checked
- Save output images to the `output/YYYYMMDD/` directory, with filenames based on the image hash from the API response.
- Optionally, save the input image alongside the output image by enabling the 'Save Input Image with output' setting. The input image will be saved in the same directory, with `_input` appended to the filename, and will be converted to match the output filetype (e.g., JPEG or PNG).
- Choose the output aspect ratio with a dropdown (default: Match input image, or select from preset ratios like 16:9, 4:3, etc.)
- Output images are displayed in a grid (gallery) when generating multiple images (batch size > 1)
- **Multiple Input Mode:** Use up to 4 images in a single prompt (see below)
- Seed handling, batch output, and more!

---

## Multiple Input Mode (Kontext Max)

**You can now use up to 4 images as input for a single prompt!**

- Switch to the "Multiple Input" tab at the top of the UI.
- Upload or paste up to 4 images (or use image URLs).
- Reference the images in your prompt (e.g., "Put the woman from image 1 in the pose from image 3").
- The app will upload your images to Fal's storage and use the returned URLs for the API call, so you can use high-resolution images without hitting size limits.
- You can leave any image slot empty; only filled images are sent.
- All other settings (prompt, batch size, aspect ratio, etc.) work as in single-image mode.

**Example prompt:**
```
The woman from image one, wearing the wrestling outfit from image 2, in the pose on her knees doing yoga from image 3
```

**How it works:**
- Each image is uploaded to Fal's file storage and referenced by URL in the API call.
- This avoids the 4MB request size limit and allows for large images.
- The Kontext Max API endpoint is used for multi-image jobs.

---

## Single Input Mode

The original single-image workflow is still available under the "Single Input" tab. This mode uses the standard Kontext API and supports all the same prompt and output options as before.

---

## Notes
- `.active`, `activity.log`, and `output/` are used for logging and output management.
- For multi-image jobs, images are uploaded to Fal's storage and not sent as base64 in the request body.
- You can mix local uploads and URLs in multi-image mode.
- If you hit any errors, check the terminal for debug output.

---

## Running

Install requirements:
```
pip install -r requirements.txt
```

Run the app:
```
python kontext-ui.py
```

Add `--verbose` for debug output:
```
python kontext-ui.py --verbose
```

### Host and Port Options
You can specify the network interface and port for the Gradio UI using `--host` and `--port`:
```
python kontext-ui.py --host 0.0.0.0 --port 7500
```
- `--host` (default: `0.0.0.0`) sets the network interface to listen on (e.g., `127.0.0.1` for localhost only).
- `--port` (default: `7500`) sets the port for the UI.
- The UI will be available at the address shown in the terminal (e.g., http://localhost:7500)
- All activity is logged to `activity.log` (rolls over to `old_activity.log` at 10MB)
- Output images are saved to the `output/YYYYMMDD/` directory if 'Save output image' is checked
- **To use the video-to-image feature:** Upload a video file, click "Extract First Frame from Video", and the first frame will be set as the input image for editing
- **Image Transformations:** After loading an input image, use the transformation dropdown and "Transform" button to:
  - Rotate the image left or right (90° increments)
  - Flip the image horizontally or vertically
  - Apply multiple transformations (transformations are cumulative and repeatable)
  - The transformed image updates in real-time and will be used for API processing
  - The transformed image updates in real-time and will be used for API processing
- **Output Aspect Ratio:** Use the dropdown in Additional Settings to match the input image's aspect ratio or select a preset (e.g., 16:9, 4:3, 1:1, etc.).
- **Batch Output:** When generating multiple images, all outputs are shown in a grid below the input image.
- **Seed Handling:** The seed input will update to the actual seed used after each run, unless 'Lock Seed' is checked.
- **Video-to-Image:** Upload a video and extract the first frame from the left column, then use it as your input image.

---

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

## API Key Setup

On first run, the app will prompt you for your Fal API key and store it in a local hidden file called `.fal_key` in the project root. This file is used for all future runs and is ignored by git (not committed to version control).

- If you want to change the key, simply delete the `.fal_key` file and restart the app.
- **Alternatively, use the `--clearkey` option to safely clear and reset your API key:**
  ```sh
  python kontext-ui.py --clearkey
  ```
  This will delete the `.fal_key` file, clear the key from memory, and prompt you to enter a new key.
- If you set the `FAL_KEY` environment variable, it will override the value in `.fal_key`.
- **Do not share your `.fal_key` file.**

You do not need to manually set environment variables unless you want to override the key for a specific session.

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
- **Image Transformations:** After loading an input image, use the transformation dropdown and "Transform" button to:
  - Rotate the image left or right (90° increments)
  - Flip the image horizontally or vertically
  - Apply multiple transformations (transformations are cumulative and repeatable)
  - The transformed image updates in real-time and will be used for API processing
- **Output Aspect Ratio:** Use the dropdown in Additional Settings to match the input image's aspect ratio or select a preset (e.g., 16:9, 4:3, 1:1, etc.).
- **Batch Output:** When generating multiple images, all outputs are shown in a grid below the input image.
- **Seed Handling:** The seed input will update to the actual seed used after each run, unless 'Lock Seed' is checked.
- **Video-to-Image:** Upload a video and extract the first frame from the left column, then use it as your input image.

## Notes
- `.active`, `activity.log`, `
