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
```