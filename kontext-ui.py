import os
import argparse
import gradio as gr
from PIL import Image
import requests
import fal_client
import io
import base64
import tempfile
import random
import re
from datetime import datetime
import csv
import shutil
import time
import sys
import cv2
from typing import List, Optional, Tuple, Union

VERBOSE = False
FAL_KEY = None

def debug(msg):
    if VERBOSE:
        print(f"[DEBUG] {msg}")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

ACTIVITY_LOG_FILE = "activity.log"
OLD_ACTIVITY_LOG_FILE = "old_activity.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB

MAX_PIXELS = 1240000  # 1.24MP
MAX_INPUT_IMAGES = 4  # Maximum number of input images for multi-image mode

# API endpoints
KONTEXT_ENDPOINT = "fal-ai/flux-pro/kontext"
KONTEXT_MAX_ENDPOINT = "fal-ai/flux-pro/kontext/max/multi"

def ensure_gitignore():
    gi_path = ".gitignore"
    lines = set()
    if os.path.exists(gi_path):
        with open(gi_path, "r") as f:
            lines = set(l.strip() for l in f)
    needed = {ACTIVITY_LOG_FILE, OLD_ACTIVITY_LOG_FILE}
    if not needed.issubset(lines):
        with open(gi_path, "a") as f:
            for n in needed:
                if n not in lines:
                    f.write(f"{n}\n")

def log_activity(prompt, output_url, nsfw_flagged):
    # Roll log if needed
    if os.path.exists(ACTIVITY_LOG_FILE) and os.path.getsize(ACTIVITY_LOG_FILE) > MAX_LOG_SIZE:
        shutil.move(ACTIVITY_LOG_FILE, OLD_ACTIVITY_LOG_FILE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ACTIVITY_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, prompt, output_url, nsfw_flagged])

def process_active_jobs_on_startup():
    jobs = list_active_jobs()
    for job in jobs:
        if len(job) != 4:
            debug(f"Skipping malformed .active line: {job}")
            continue
        request_id, ts, prompt, input_url = job
        debug(f"Resuming job {request_id} from .active...")
        try:
            status = fal_client.status("fal-ai/flux-pro/kontext", request_id, with_logs=False)
            if hasattr(status, 'status') and getattr(status, 'status', None) == "COMPLETED":
                result = fal_client.result("fal-ai/flux-pro/kontext", request_id)
                images = result.get('images')
                urls_out = []
                images_out = []
                if images and isinstance(images, list):
                    for img in images:
                        if 'url' in img:
                            output_url = img['url']
                            debug(f"Fetching output image from {output_url}")
                            response = requests.get(output_url)
                            images_out.append(Image.open(io.BytesIO(response.content)))
                            urls_out.append(output_url)
                            log_activity(prompt, output_url, False)
                # Save images to output dir
                if images_out and urls_out:
                    outdir = os.path.join("output", datetime.now().strftime("%Y%m%d"))
                    os.makedirs(outdir, exist_ok=True)
                    for img, out_url in zip(images_out, urls_out):
                        fal_hash = extract_fal_hash(out_url)
                        if fal_hash:
                            ext = os.path.splitext(out_url)[1]
                            if fal_hash.endswith(ext):
                                filename = os.path.join(outdir, fal_hash)
                            else:
                                filename = os.path.join(outdir, f"{fal_hash}{ext}")
                            img.save(filename)
                            debug(f"Saved output image: {filename}")
                remove_active_job(request_id)
        except Exception as e:
            debug(f"Error resuming job {request_id}: {e}")

def call_kontext_max(prompt: str, image_urls: List[str], safety: int, seed: int, guidance_scale: float, 
                     num_images: int, output_format: str, raw: bool, output_aspect: str, 
                     image_size: Optional[dict], aspect_ratio: Optional[str], 
                     image_prompt_strength: float, num_inference_steps: int) -> Tuple[List[Image.Image], List[str]]:
    """Call the Kontext Max API with multiple input images."""
    debug(f"call_kontext_max starting with {len(image_urls)} input images")
    payload = {
        "prompt": prompt,
        "image_urls": image_urls,
        "safety_tolerance": str(safety),
        "seed": int(seed),
        "guidance_scale": float(guidance_scale),
        "num_images": int(num_images),
        "output_format": output_format,
        "raw": bool(raw),
        "image_prompt_strength": float(image_prompt_strength),
        "num_inference_steps": int(num_inference_steps)
    }
    if image_size:
        payload["image_size"] = image_size
    elif aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio

    if VERBOSE:
        payload_log = payload.copy()
        payload_log["image_urls"] = [url if not url.startswith("data:image/") else "(base64 image)" for url in image_urls]
        print(f"[VERBOSE] Upload payload: {payload_log}")

    api_result_holder = {}
    def on_queue_update(update):
        debug(f"Queue update: {update}")
    
    try:
        debug(f"Calling fal_client.subscribe for Kontext Max...")
        result = fal_client.subscribe(
            KONTEXT_MAX_ENDPOINT,
            arguments=payload,
            with_logs=VERBOSE,
            on_queue_update=on_queue_update if VERBOSE else None,
        )
        debug(f"API response received")
    except Exception as e:
        debug(f"API call failed with error: {type(e).__name__}: {str(e)}")
        print(f"[ERROR] API call failed: {type(e).__name__}: {str(e)}")
        if "unauthorized" in str(e).lower() or "401" in str(e):
            print("[ERROR] Authentication failed. Your FAL_KEY may be invalid.")
            print("[ERROR] Delete the .fal_key file and restart to enter a new key.")
        raise
    
    debug(f"API response: {result}")
    api_result_holder['result'] = result
    images_out = []
    urls_out = []
    if isinstance(result, dict):
        images = result.get('images')
        if images and isinstance(images, list):
            for img in images:
                if 'url' in img:
                    output_url = img['url']
                    debug(f"Fetching output image from {output_url}")
                    response = requests.get(output_url)
                    images_out.append(Image.open(io.BytesIO(response.content)))
                    urls_out.append(output_url)
    
    # Store the last API result for NSFW checking
    call_kontext_max.last_api_result = api_result_holder['result']
    return images_out, urls_out

def call_kontext(prompt: str, image_url: str, safety: int, seed: int, guidance_scale: float, 
                 num_images: int, output_format: str, raw: bool, output_aspect: str, 
                 image_size: Optional[dict], aspect_ratio: Optional[str], 
                 image_prompt_strength: float, num_inference_steps: int) -> Tuple[List[Image.Image], List[str]]:
    """Call the standard Kontext API with a single input image."""
    debug(f"call_kontext starting with FAL_KEY in env: {bool(os.environ.get('FAL_KEY'))}")
    payload = {
        "prompt": prompt,
        "image_url": image_url,
        "safety_tolerance": str(safety),
        "seed": int(seed),
        "guidance_scale": float(guidance_scale),
        "num_images": int(num_images),
        "output_format": output_format,
        "raw": bool(raw),
        "image_prompt_strength": float(image_prompt_strength),
        "num_inference_steps": int(num_inference_steps)
    }
    if image_size:
        payload["image_size"] = image_size
    elif aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio

    if VERBOSE:
        payload_log = payload.copy()
        if isinstance(payload_log.get("image_url"), str) and payload_log["image_url"].startswith("data:image/"):
            payload_log["image_url"] = "(base64 image)"
        print(f"[VERBOSE] Upload payload: {payload_log}")

    api_result_holder = {}
    def on_queue_update(update):
        debug(f"Queue update: {update}")
    
    try:
        debug(f"Calling fal_client.subscribe...")
        result = fal_client.subscribe(
            KONTEXT_ENDPOINT,
            arguments=payload,
            with_logs=VERBOSE,
            on_queue_update=on_queue_update if VERBOSE else None,
        )
        debug(f"API response received")
    except Exception as e:
        debug(f"API call failed with error: {type(e).__name__}: {str(e)}")
        print(f"[ERROR] API call failed: {type(e).__name__}: {str(e)}")
        if "unauthorized" in str(e).lower() or "401" in str(e):
            print("[ERROR] Authentication failed. Your FAL_KEY may be invalid.")
            print("[ERROR] Delete the .fal_key file and restart to enter a new key.")
        raise
    
    debug(f"API response: {result}")
    api_result_holder['result'] = result
    images_out = []
    urls_out = []
    if isinstance(result, dict):
        images = result.get('images')
        if images and isinstance(images, list):
            for img in images:
                if 'url' in img:
                    output_url = img['url']
                    debug(f"Fetching output image from {output_url}")
                    response = requests.get(output_url)
                    images_out.append(Image.open(io.BytesIO(response.content)))
                    urls_out.append(output_url)
    
    # Store the last API result for NSFW checking
    call_kontext.last_api_result = api_result_holder['result']
    return images_out, urls_out

def resize_to_max_pixels(img: Image.Image, max_pixels=MAX_PIXELS) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    aspect = w / h
    new_h = int((max_pixels / aspect) ** 0.5)
    new_w = int(new_h * aspect)
    debug(f"Resizing image from {w}x{h} to {new_w}x{new_h} to fit under {max_pixels} pixels.")
    return img.resize((new_w, new_h), Image.LANCZOS)

def image_to_data_uri(img: Image.Image, format: str = "PNG") -> str:
    img = resize_to_max_pixels(img)
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format.lower()};base64,{base64_str}"

def process(prompt: str, raw: bool, images: Union[Image.Image, List[Image.Image]], safety: int, 
           seed: int, guidance_scale: float, num_images: int, output_format: str, 
           output_aspect: str, image_size: Optional[dict], aspect_ratio: Optional[str], 
           image_prompt_strength: float, num_inference_steps: int, save_output: bool, 
           save_input: bool, is_multi_mode: bool = False) -> Tuple[List[Image.Image], str]:
    """Process images with the Kontext API, supporting both single and multi-image modes."""
    if is_multi_mode:
        if not isinstance(images, list):
            images = [images] if images else []
        # Filter out None values and convert images to URLs
        image_urls = []
        valid_images = []
        temp_files = []
        for img in images:
            if isinstance(img, Image.Image):
                # Resize image before uploading
                img = resize_to_max_pixels(img)
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    img.save(tmp, format='PNG')
                    temp_path = tmp.name
                temp_files.append(temp_path)
                # Upload to Fal storage
                try:
                    url = fal_client.upload_file(temp_path)
                    image_urls.append(url)
                    valid_images.append(img)
                except Exception as e:
                    debug(f"Failed to upload image to Fal storage: {e}")
            elif isinstance(img, str) and img.startswith("http"):
                image_urls.append(img)
                valid_images.append(None)  # We don't have the actual image for HTTP URLs
        # Clean up temp files
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass
        if not image_urls:
            return [], "No valid input images provided."
        result = call_kontext_max(prompt, image_urls, safety, seed, guidance_scale, num_images, 
                                output_format, raw, output_aspect, image_size, aspect_ratio, 
                                image_prompt_strength, num_inference_steps)
    else:
        if isinstance(images, list):
            images = images[0] if images else None
        if isinstance(images, Image.Image):
            images = resize_to_max_pixels(images)
            data_uri = image_to_data_uri(images, format='PNG')
            url = data_uri
        elif isinstance(images, str) and images.startswith("http"):
            url = images
        else:
            return [], "No valid input image provided."
        
        result = call_kontext(prompt, url, safety, seed, guidance_scale, num_images, 
                            output_format, raw, output_aspect, image_size, aspect_ratio, 
                            image_prompt_strength, num_inference_steps)
    
    out_imgs, out_urls = result
    nsfw_flags = []
    api_result = (call_kontext_max.last_api_result if is_multi_mode else call_kontext.last_api_result)
    if api_result and isinstance(api_result, dict):
        has_nsfw = api_result.get('has_nsfw_concepts')
        if has_nsfw:
            nsfw_flags = list(has_nsfw)
    
    # Log all outputs, even if NSFW
    for i, out_url in enumerate(out_urls):
        nsfw_flagged = nsfw_flags[i] if i < len(nsfw_flags) else False
        log_activity(prompt, out_url, nsfw_flagged)
    
    # Save images if requested, skipping those flagged as NSFW
    saved_any = False
    if save_output and out_imgs and out_urls:
        outdir = os.path.join("output", datetime.now().strftime("%Y%m%d"))
        os.makedirs(outdir, exist_ok=True)
        for i, (img, out_url) in enumerate(zip(out_imgs, out_urls)):
            nsfw_flagged = nsfw_flags[i] if i < len(nsfw_flags) else False
            fal_hash = extract_fal_hash(out_url)
            if fal_hash:
                ext = os.path.splitext(out_url)[1]
                if not ext:
                    ext = f'.{output_format}'
                if not fal_hash.endswith(ext):
                    filename = os.path.join(outdir, f"{fal_hash}{ext}")
                else:
                    filename = os.path.join(outdir, fal_hash)
                if nsfw_flagged:
                    debug(f"Skipping save for {filename} due to NSFW block.")
                    print(f"Image blocked for NSFW, not saved: {filename}")
                    continue
                try:
                    debug(f"Saving output image to: {filename}")
                    img.save(filename)
                    debug(f"Saved output image: {filename}")
                    saved_any = True
                    # Save input images if requested
                    if save_input and valid_images:
                        for j, input_img in enumerate(valid_images):
                            if input_img is not None:  # Skip HTTP URLs
                                input_filename = filename.replace(ext, f"_input{j+1}{ext}")
                                try:
                                    input_img.save(input_filename, format=output_format.upper())
                                    debug(f"Saved input image {j+1}: {input_filename}")
                                except Exception as e:
                                    debug(f"Failed to save input image {j+1} to {input_filename}: {e}")
                except Exception as e:
                    debug(f"Failed to save output image to {filename}: {e}")
    
    # Print URLs to CLI
    for out_url in out_urls:
        print(f"Output URL: {out_url}")
    
    # Info box: if all images are NSFW, show blocked message; else show URLs
    all_nsfw = all((nsfw_flags[i] if i < len(nsfw_flags) else False) for i in range(len(out_urls))) if out_urls else False
    if all_nsfw:
        info_md = f"**Prompt:** {prompt}\n\n**Seed:** {seed}\n\n**Blocked for NSFW, image not saved.**"
        return [], info_md
    
    info_md = f"**Prompt:** {prompt}\n\n**Seed:** {seed}\n\n**Output URL(s):**\n" + "\n".join(out_urls if out_urls else ['N/A'])
    return out_imgs, info_md

def extract_fal_hash(url):
    # Extract everything after /files/<any-animal>/ in the URL
    m = re.search(r"/files/[^/]+/([^/?#]+)", url)
    if m:
        return m.group(1)
    return None

def ensure_env_var(var_name, prompt_text):
    debug(f"ensure_env_var called for {var_name}")
    value = os.environ.get(var_name)
    if value:
        debug(f"{var_name} found in environment: {'*' * min(4, len(value))}...{value[-4:] if len(value) > 4 else ''}")
        return value
    debug(f"{var_name} not found in environment")
    
    key_file = ".fal_key"
    debug(f"Checking for {key_file} file...")
    if os.path.exists(key_file):
        debug(f"{key_file} file exists, attempting to read...")
        with open(key_file, "r") as f:
            value = f.read().strip()
            if value:
                debug(f"Found value in {key_file}: {'*' * min(4, len(value))}...{value[-4:] if len(value) > 4 else ''}")
                os.environ[var_name] = value
                return value
            else:
                debug(f"{key_file} file exists but is empty")
    else:
        debug(f"{key_file} file does not exist")
    
    # Prompt user for the value
    print(f"[Kontext UI] {var_name} not found in environment or .fal_key file.")
    value = input(f"Please enter your {prompt_text}: ").strip()
    os.environ[var_name] = value
    with open(key_file, "w") as f:
        f.write(value)
    print(f"[Kontext UI] {var_name} saved to {key_file} for future runs.")
    debug(f"New {var_name} saved to {key_file}")
    return value

def extract_first_frame_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise ValueError("Could not read first frame from video.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = resize_to_max_pixels(img)
    return img

def transform_image(img, transformation):
    """Apply transformation to image"""
    if not isinstance(img, Image.Image):
        return img
    
    if transformation == "Rotate Input Image Left (90°)":
        return img.rotate(90, expand=True)
    elif transformation == "Rotate Input Image Right (90°)":
        return img.rotate(-90, expand=True)
    elif transformation == "Flip Input Image Horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif transformation == "Flip Input Image Vertical":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return img

def clear_fal_key():
    """Clear FAL_KEY from environment and delete .fal_key file"""
    debug("clear_fal_key called")
    
    # Ask for confirmation
    print("\n[WARNING] This will:")
    print("  1. Delete the .fal_key file")
    print("  2. Clear FAL_KEY from the current environment")
    print("  3. Prompt you to enter a new key")
    response = input("\nAre you sure you want to clear the FAL key? (yes/no): ").strip().lower()
    
    if response != "yes":
        print("Operation cancelled.")
        return False
    
    # Clear from environment
    if "FAL_KEY" in os.environ:
        del os.environ["FAL_KEY"]
        debug("FAL_KEY cleared from environment")
        print("✓ FAL_KEY cleared from environment")
    else:
        debug("FAL_KEY was not in environment")
    
    # Delete .fal_key file
    key_file = ".fal_key"
    if os.path.exists(key_file):
        try:
            os.remove(key_file)
            debug(f"{key_file} file deleted")
            print(f"✓ {key_file} file deleted")
        except Exception as e:
            print(f"[ERROR] Failed to delete {key_file}: {e}")
            return False
    else:
        debug(f"{key_file} file did not exist")
        print(f"  {key_file} file did not exist")
    
    print("\nFAL key cleared successfully.")
    return True

def main():
    global VERBOSE, FAL_KEY
    parser = argparse.ArgumentParser(description="Kontext UI - Fal Kontext Evaluation UI")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to serve on')
    parser.add_argument('--port', type=int, default=7500, help='Port to serve on')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output')
    parser.add_argument('--clearkey', action='store_true', help='Clear FAL_KEY from environment and file, then prompt for new key')
    args = parser.parse_args()
    VERBOSE = args.verbose

    # Handle --clearkey argument
    if args.clearkey:
        if clear_fal_key():
            print("\nNow prompting for new FAL key...")
        else:
            print("\nExiting without changes.")
            sys.exit(0)
    
    # Initialize FAL_KEY
    FAL_KEY = ensure_env_var("FAL_KEY", "Fal API key")
    debug(f"FAL_KEY set: {bool(FAL_KEY)}")
    
    # Ensure fal_client uses the FAL_KEY
    if FAL_KEY:
        os.environ["FAL_KEY"] = FAL_KEY
        debug(f"FAL_KEY explicitly set in os.environ")
        try:
            fal_client.api_key = FAL_KEY
            debug(f"fal_client.api_key set directly")
        except AttributeError:
            debug(f"fal_client doesn't have api_key attribute, relying on environment variable")
    else:
        print("[ERROR] Failed to set FAL_KEY. Exiting.")
        sys.exit(1)

    # Randomize default seed on UI load
    default_seed = random.randint(0, 2**32 - 1)

    with gr.Blocks() as demo:
        gr.Markdown("# Kontext UI - Open Source Fal Kontext Demo")
        gr.Markdown("""
<style>
#run-btn {
    background-color: #FFD600 !important;
    color: #222 !important;
    border: 1px solid #e6c200 !important;
    height: 64px !important;
    font-size: 1.2em !important;
    min-height: 64px !important;
    max-height: 128px !important;
    line-height: 2.5em !important;
}
#transform-btn {
    height: 48px !important;
    min-height: 48px !important;
    font-size: 1em !important;
    line-height: 1.5em !important;
}
#prompt-row {
    background: #1a3321 !important;
    border-radius: 8px;
    padding: 8px 0 8px 0;
}
.batch-size-row {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
}
.batch-size-label {
    margin-right: 8px;
    font-weight: 500;
}
.batch-size-input input {
    width: 2.5em !important;
    min-width: 2.5em !important;
    max-width: 3.5em !important;
    text-align: center;
}
.seed-lock-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.seed-label {
    flex: 4 1 0%;
    font-weight: 500;
    display: flex;
    align-items: center;
}
.seed-input input {
    width: 7em !important;
    min-width: 7em !important;
    margin-left: 8px;
}
.lock-seed-col {
    flex: 1 1 0%;
    display: flex;
    align-items: center;
    justify-content: flex-end;
}
.transform-row {
    align-items: center !important;
    display: flex !important;
}
.transform-row > div {
    display: flex !important;
    align-items: center !important;
}
.transform-row .gradio-dropdown {
    margin-top: 0 !important;
}
.transform-row .gradio-dropdown label {
    display: none !important;
}
#input-image {
    position: relative;
}
#input-image:focus {
    outline: 2px solid #FFD600;
}
.paste-info {
    font-size: 0.9em;
    color: #888;
    margin-top: 8px;
    text-align: center;
}
.multi-input-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-bottom: 8px;
}
.multi-input-grid .gradio-image {
    aspect-ratio: 1;
    object-fit: cover;
}
.multi-input-grid .gradio-image img {
    aspect-ratio: 1;
    object-fit: cover;
}
</style>

<script>
// Simple enhancement for better UX
function setupSimpleEnhancements() {
    setTimeout(() => {
        const imageContainers = document.querySelectorAll('#input-image, .multi-input-grid .gradio-image');
        imageContainers.forEach(container => {
            // Add better visual feedback for drag and drop
            let dragCounter = 0;
            
            ['dragenter', 'dragover'].forEach(eventName => {
                container.addEventListener(eventName, (e) => {
                    if (e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
                        e.preventDefault();
                        dragCounter++;
                        container.style.border = '3px dashed #FFD600';
                        container.style.backgroundColor = 'rgba(255, 214, 0, 0.1)';
                    }
                });
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                container.addEventListener(eventName, (e) => {
                    if (e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
                        dragCounter--;
                        if (dragCounter <= 0) {
                            dragCounter = 0;
                            container.style.border = '';
                            container.style.backgroundColor = '';
                        }
                    }
                });
            });
        });
        
        console.log('Simple drag/drop enhancements loaded');
    }, 1000);
}

// Initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupSimpleEnhancements);
} else {
    setupSimpleEnhancements();
}
</script>
""")
        with gr.Tabs() as tabs:
            with gr.TabItem("Single Input"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            with gr.Column(scale=1):
                                num_images = gr.Number(label="Batch Size", value=1, precision=0, interactive=True, minimum=1, maximum=4)
                            with gr.Column(scale=1):
                                seed = gr.Number(label="Seed", value=default_seed, precision=0, interactive=True)
                            with gr.Column(scale=1):
                                lock_seed = gr.Checkbox(label="Lock Seed", value=False)
                        run_btn = gr.Button("Run", elem_id="run-btn", variant="primary")
                        with gr.Row(elem_id="prompt-row"):
                            prompt = gr.Textbox(label="Prompt", value="", lines=3, elem_id="prompt-box")
                        raw = gr.Checkbox(label="Disable prompt enhancement", value=False)
                        with gr.Accordion("Additional Settings", open=True):
                            image_prompt_strength = gr.Slider(label="Prompt Strength", minimum=0.0, maximum=1.0, step=0.01, value=0.1)
                            num_inference_steps = gr.Slider(label="Steps", minimum=10, maximum=100, step=1, value=28)
                            guidance_scale = gr.Slider(label="CFG (Guidance Scale)", minimum=1.0, maximum=10.0, step=0.1, value=3.5)
                            output_format = gr.Dropdown(label="Output Format", choices=["jpeg", "png"], value="jpeg")
                            aspect_ratio_choices = [
                                "Match input image",
                                "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"
                            ]
                            output_aspect = gr.Dropdown(label="Output Aspect Ratio", choices=aspect_ratio_choices, value="Match input image")
                            safety = gr.Slider(label="Safety Tolerance (1=Strict, 6=Permissive)", minimum=1, maximum=6, step=1, value=5)
                            save_input = gr.Checkbox(label="Save Input Image with output", value=False)
                        video_upload = gr.File(label="Upload Video (extract first frame)", file_types=[".mp4", ".mov", ".avi", ".webm"], type="filepath")
                        extract_btn = gr.Button("Extract First Frame from Video")
                    with gr.Column():
                        image = gr.Image(label="Input Image", type="pil", height=512, show_label=True, elem_id="input-image", sources=["upload", "webcam", "clipboard"])
                        # Image transformation controls
                        with gr.Row(elem_classes=["transform-row"]):
                            transform_dropdown = gr.Dropdown(
                                label="",
                                choices=["Rotate Input Image Left (90°)", "Rotate Input Image Right (90°)", "Flip Input Image Horizontal", "Flip Input Image Vertical"], 
                                value="Rotate Input Image Left (90°)"
                            )
                            transform_btn = gr.Button("Transform", elem_id="transform-btn")
                        after = gr.Gallery(label="Output Images", show_label=True, height=512, elem_id="output-image", columns=[2])
                        save_output = gr.Checkbox(label="Save output image", value=True)
                        info_box = gr.Markdown("", elem_id="output-info")

            with gr.TabItem("Multiple Input"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            with gr.Column(scale=1):
                                num_images_multi = gr.Number(label="Batch Size", value=1, precision=0, interactive=True, minimum=1, maximum=4)
                            with gr.Column(scale=1):
                                seed_multi = gr.Number(label="Seed", value=default_seed, precision=0, interactive=True)
                            with gr.Column(scale=1):
                                lock_seed_multi = gr.Checkbox(label="Lock Seed", value=False)
                        run_btn_multi = gr.Button("Run", elem_id="run-btn", variant="primary")
                        with gr.Row(elem_id="prompt-row"):
                            prompt_multi = gr.Textbox(label="Prompt", value="", lines=3, elem_id="prompt-box")
                        raw_multi = gr.Checkbox(label="Disable prompt enhancement", value=False)
                        with gr.Accordion("Additional Settings", open=True):
                            image_prompt_strength_multi = gr.Slider(label="Prompt Strength", minimum=0.0, maximum=1.0, step=0.01, value=0.1)
                            num_inference_steps_multi = gr.Slider(label="Steps", minimum=10, maximum=100, step=1, value=28)
                            guidance_scale_multi = gr.Slider(label="CFG (Guidance Scale)", minimum=1.0, maximum=10.0, step=0.1, value=3.5)
                            output_format_multi = gr.Dropdown(label="Output Format", choices=["jpeg", "png"], value="jpeg")
                            aspect_ratio_choices = [
                                "Match input image",
                                "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"
                            ]
                            output_aspect_multi = gr.Dropdown(label="Output Aspect Ratio", choices=aspect_ratio_choices, value="Match input image")
                            safety_multi = gr.Slider(label="Safety Tolerance (1=Strict, 6=Permissive)", minimum=1, maximum=6, step=1, value=5)
                            save_input_multi = gr.Checkbox(label="Save Input Images with output", value=False)
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Input Images (up to 4)")
                                gr.Markdown("Upload images or paste from clipboard. You can use up to 4 images as input.")
                                with gr.Row(elem_classes=["multi-input-grid"]):
                                    images_multi = [
                                        gr.Image(label=f"Input Image {i+1}", type="pil", height=256, show_label=True, 
                                               sources=["upload", "webcam", "clipboard"], elem_id=f"input-image-{i}")
                                        for i in range(MAX_INPUT_IMAGES)
                                    ]
                        after_multi = gr.Gallery(label="Output Images", show_label=True, height=512, elem_id="output-image", columns=[2])
                        save_output_multi = gr.Checkbox(label="Save output image", value=True)
                        info_box_multi = gr.Markdown("", elem_id="output-info")

        last_seed = {"value": default_seed}
        def run_all(prompt, raw, image, safety, seed, lock_seed, guidance_scale, num_images, output_format, output_aspect, image_prompt_strength, num_inference_steps, save_output, save_input):
            if lock_seed:
                use_seed = int(seed)
                new_seed = use_seed
            else:
                use_seed = random.randint(0, 2**32 - 1)
                new_seed = use_seed
            # Determine image_size or aspect_ratio for API
            image_size = None
            aspect_ratio = None
            resized_dims = None
            if output_aspect == "Match input image" and isinstance(image, Image.Image):
                # Use resized image dimensions, capped at 1.24MP
                resized = resize_to_max_pixels(image)
                resized_dims = resized.size
                image_size = {"width": resized_dims[0], "height": resized_dims[1]}
            elif output_aspect != "Match input image":
                aspect_ratio = output_aspect
            out_imgs, info_md = process(prompt, raw, image, int(safety), use_seed, float(guidance_scale), int(num_images), 
                                      output_format, output_aspect, image_size, aspect_ratio, float(image_prompt_strength), 
                                      int(num_inference_steps), save_output, save_input, is_multi_mode=False)
            return out_imgs, info_md, gr.update(value=new_seed)

        def run_all_multi(prompt, raw, img1, img2, img3, img4, safety, seed, lock_seed, guidance_scale, num_images, output_format, 
                         output_aspect, image_prompt_strength, num_inference_steps, save_output, save_input):
            if lock_seed:
                use_seed = int(seed)
                new_seed = use_seed
            else:
                use_seed = random.randint(0, 2**32 - 1)
                new_seed = use_seed
            # Determine image_size or aspect_ratio for API
            image_size = None
            aspect_ratio = None
            if output_aspect != "Match input image":
                aspect_ratio = output_aspect
            # Collect images and filter out None values
            images = [img1, img2, img3, img4]
            valid_images = [img for img in images if img is not None]
            if not valid_images:
                return [], "No valid input images provided.", gr.update(value=new_seed)
            # Use first image's dimensions if matching input
            if output_aspect == "Match input image" and isinstance(valid_images[0], Image.Image):
                resized = resize_to_max_pixels(valid_images[0])
                resized_dims = resized.size
                image_size = {"width": resized_dims[0], "height": resized_dims[1]}
            out_imgs, info_md = process(prompt, raw, valid_images, int(safety), use_seed, float(guidance_scale), 
                                      int(num_images), output_format, output_aspect, image_size, aspect_ratio, 
                                      float(image_prompt_strength), int(num_inference_steps), save_output, save_input, 
                                      is_multi_mode=True)
            return out_imgs, info_md, gr.update(value=new_seed)

        # Run button click handlers
        run_btn.click(run_all, inputs=[prompt, raw, image, safety, seed, lock_seed, guidance_scale, num_images, 
                                     output_format, output_aspect, image_prompt_strength, num_inference_steps, 
                                     save_output, save_input], outputs=[after, info_box, seed])
        
        run_btn_multi.click(run_all_multi, inputs=[prompt_multi, raw_multi, *images_multi, safety_multi, seed_multi, 
                                                 lock_seed_multi, guidance_scale_multi, num_images_multi, 
                                                 output_format_multi, output_aspect_multi, image_prompt_strength_multi, 
                                                 num_inference_steps_multi, save_output_multi, save_input_multi], 
                           outputs=[after_multi, info_box_multi, seed_multi])

        # Prompt box: run on Ctrl+Enter
        prompt.submit(run_all, inputs=[prompt, raw, image, safety, seed, lock_seed, guidance_scale, num_images, 
                                     output_format, output_aspect, image_prompt_strength, num_inference_steps, 
                                     save_output, save_input], outputs=[after, info_box, seed], queue=True, preprocess=True)
        
        prompt_multi.submit(run_all_multi, inputs=[prompt_multi, raw_multi, *images_multi, safety_multi, seed_multi, 
                                                 lock_seed_multi, guidance_scale_multi, num_images_multi, 
                                                 output_format_multi, output_aspect_multi, image_prompt_strength_multi, 
                                                 num_inference_steps_multi, save_output_multi, save_input_multi], 
                           outputs=[after_multi, info_box_multi, seed_multi], queue=True, preprocess=True)

        # Video upload: extract first frame and set as image input
        def handle_video_extract(video_path):
            if not video_path:
                return gr.update()
            try:
                img = extract_first_frame_from_video(video_path)
                return img
            except Exception as e:
                print(f"[ERROR] Could not extract frame: {e}")
                return gr.update()

        # Transform button handler
        def handle_transform(current_image, transformation):
            if current_image is None:
                return gr.update()
            try:
                transformed_img = transform_image(current_image, transformation)
                return transformed_img
            except Exception as e:
                print(f"[ERROR] Could not transform image: {e}")
                return gr.update()

        extract_btn.click(handle_video_extract, inputs=[video_upload], outputs=[image])
        transform_btn.click(handle_transform, inputs=[image, transform_dropdown], outputs=[image])

    # Ensure .gitignore is updated
    ensure_gitignore()
    demo.launch(server_name=args.host, server_port=args.port, share=False)

if __name__ == "__main__":
    main() 
