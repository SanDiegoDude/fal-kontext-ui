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

VERBOSE = False

def debug(msg):
    if VERBOSE:
        print(f"[DEBUG] {msg}")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IMG_API_URL = "https://api.imgbb.com/1/upload"
IMG_EXPIRATION_TIME = 300  # 5 minutes

ACTIVE_JOBS_FILE = ".active"
ACTIVITY_LOG_FILE = "activity.log"
OLD_ACTIVITY_LOG_FILE = "old_activity.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB

MAX_PIXELS = 1240000  # 1.24MP

def get_imgbb_api_key():
    key = os.environ.get("IMGBB_API_KEY")
    if not key:
        raise EnvironmentError("IMGBB_API_KEY not set in environment.")
    return key

def upload_image_to_imgbb(local_image_path: str) -> str:
    imgbb_key = get_imgbb_api_key()
    filename = os.path.basename(local_image_path)
    with open(local_image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    url = f"{IMG_API_URL}?key={imgbb_key}&expiration={IMG_EXPIRATION_TIME}"
    payload = {"image": base64_image}
    debug(f"Uploading {filename} to ImgBB...")
    response = requests.post(url, data=payload)
    result = response.json()
    if result.get("success"):
        image_url = result["data"]["url"]
        debug(f"{filename} uploaded to ImgBB: {image_url}")
        return image_url
    else:
        error_message = result.get("error", {}).get("message", "Unknown error")
        raise ValueError(f"ImgBB upload error: {error_message}")

def ensure_gitignore():
    gi_path = ".gitignore"
    lines = set()
    if os.path.exists(gi_path):
        with open(gi_path, "r") as f:
            lines = set(l.strip() for l in f)
    needed = {ACTIVE_JOBS_FILE, ACTIVITY_LOG_FILE, OLD_ACTIVITY_LOG_FILE}
    if not needed.issubset(lines):
        with open(gi_path, "a") as f:
            for n in needed:
                if n not in lines:
                    f.write(f"{n}\n")

def append_active_job(request_id, prompt, input_url):
    with open(ACTIVE_JOBS_FILE, "a") as f:
        f.write(f"{request_id},{int(time.time())},{prompt.replace(',', ' ')},{input_url}\n")

def remove_active_job(request_id):
    if not os.path.exists(ACTIVE_JOBS_FILE):
        return
    with open(ACTIVE_JOBS_FILE, "r") as f:
        lines = f.readlines()
    with open(ACTIVE_JOBS_FILE, "w") as f:
        for line in lines:
            if not line.startswith(f"{request_id},"):
                f.write(line)

def list_active_jobs():
    if not os.path.exists(ACTIVE_JOBS_FILE):
        return []
    with open(ACTIVE_JOBS_FILE, "r") as f:
        return [line.strip().split(",", 3) for line in f if line.strip()]

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

def call_kontext(prompt, image_url, safety, seed, guidance_scale, num_images, output_format, raw, output_aspect, image_size, aspect_ratio, image_prompt_strength, num_inference_steps):
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
    # Verbose logging: print payload with base64 image replaced
    if VERBOSE:
        payload_log = payload.copy()
        if isinstance(payload_log.get("image_url"), str) and payload_log["image_url"].startswith("data:image/"):
            payload_log["image_url"] = "(base64 image)"
        print(f"[VERBOSE] Upload payload: {payload_log}")
    # Note: The API currently only supports one input image (image_url). Multi-image input is not supported as of now.
    job_id_holder = {}
    api_result_holder = {}
    def on_enqueue(request_id):
        debug(f"Request enqueued with ID: {request_id}")
        append_active_job(request_id, prompt, image_url)
        job_id_holder['id'] = request_id
    def on_queue_update(update):
        debug(f"Queue update: {update}")
    result = fal_client.subscribe(
        "fal-ai/flux-pro/kontext",
        arguments=payload,
        with_logs=VERBOSE,
        on_enqueue=on_enqueue,
        on_queue_update=on_queue_update if VERBOSE else None,
    )
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
    # Remove from .active if job_id is known
    if 'id' in job_id_holder:
        remove_active_job(job_id_holder['id'])
    # Store the last API result for NSFW checking
    call_kontext.last_api_result = api_result_holder['result']
    print(f"[DEBUG] out_imgs type: {type(images_out)}, length: {len(images_out) if hasattr(images_out, '__len__') else 'N/A'}")
    print(f"[DEBUG] out_urls type: {type(urls_out)}, length: {len(urls_out) if hasattr(urls_out, '__len__') else 'N/A'}")
    print(f"[DEBUG] out_imgs contents: {images_out}")
    print(f"[DEBUG] out_urls contents: {urls_out}")
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

def process(prompt, raw, image, safety, seed, guidance_scale, num_images, output_format, output_aspect, image_size, aspect_ratio, image_prompt_strength, num_inference_steps, save_output, save_input):
    if isinstance(image, Image.Image):
        # Resize before encoding
        image = resize_to_max_pixels(image)
        data_uri = image_to_data_uri(image, format='PNG')
        url = data_uri
    elif isinstance(image, str) and image.startswith("http"):
        url = image
    else:
        return [], ""
    result = call_kontext(prompt, url, safety, seed, guidance_scale, num_images, output_format, raw, output_aspect, image_size, aspect_ratio, image_prompt_strength, num_inference_steps)
    out_imgs, out_urls = result
    nsfw_flags = []
    if hasattr(call_kontext, 'last_api_result'):
        api_result = call_kontext.last_api_result
        has_nsfw = api_result.get('has_nsfw_concepts') if isinstance(api_result, dict) else None
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
            print(f"[DEBUG] About to extract fal_hash from out_url: {out_url}")
            fal_hash = extract_fal_hash(out_url)
            print(f"[DEBUG] Result of extract_fal_hash: {fal_hash}")
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
                    print(f"[DEBUG] Attempting to save output image to: {filename}")
                    img.save(filename)
                    debug(f"Saved output image: {filename}")
                    print(f"[DEBUG] Saved output image: {filename}")
                    saved_any = True
                    # Save input image if requested
                    if save_input:
                        print(f"[DEBUG] save_input is True. image type: {type(image)}, value: {image}")
                        if isinstance(image, Image.Image):
                            input_filename = filename.replace(ext, f"_input{ext}")
                            try:
                                print(f"[DEBUG] Attempting to save input image to: {input_filename}")
                                image.save(input_filename, format=output_format.upper())
                                debug(f"Saved input image: {input_filename}")
                                print(f"[DEBUG] Saved input image: {input_filename}")
                            except Exception as e:
                                debug(f"Failed to save input image to {input_filename}: {e}")
                                print(f"[DEBUG] Failed to save input image to {input_filename}: {e}")
                        else:
                            print(f"[DEBUG] image is not a PIL.Image.Image, skipping input save.")
                except Exception as e:
                    debug(f"Failed to save output image to {filename}: {e}")
                    print(f"[DEBUG] Failed to save output image to {filename}: {e}")
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
    value = os.environ.get(var_name)
    if value:
        return value
    key_file = ".fal_key"
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            value = f.read().strip()
            if value:
                os.environ[var_name] = value
                return value
    # Prompt user for the value
    print(f"[Kontext UI] {var_name} not found in environment or .fal_key file.")
    value = input(f"Please enter your {prompt_text}: ").strip()
    os.environ[var_name] = value
    with open(key_file, "w") as f:
        f.write(value)
    print(f"[Kontext UI] {var_name} saved to {key_file} for future runs.")
    return value

# Only check for FAL_KEY now
FAL_KEY = ensure_env_var("FAL_KEY", "Fal API key")

def extract_first_frame_from_video(video_path, rotate_degrees=0):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise ValueError("Could not read first frame from video.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    if rotate_degrees in [90, 180, 270]:
        img = img.rotate(-rotate_degrees, expand=True)  # PIL rotates counterclockwise, so negative for clockwise
    img = resize_to_max_pixels(img)
    return img

def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="Kontext UI - Fal Kontext Evaluation UI")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to serve on')
    parser.add_argument('--port', type=int, default=7500, help='Port to serve on')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output')
    args = parser.parse_args()
    VERBOSE = args.verbose

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
</style>
""")
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
                rotate_choices = [0, 90, 180, 270]
                rotate_dropdown = gr.Dropdown(label="Rotate Frame Degrees", choices=rotate_choices, value=0)
                extract_btn = gr.Button("Extract First Frame from Video")
            with gr.Column():
                image = gr.Image(label="Input Image", type="pil", height=512, show_label=True, elem_id="input-image")
                after = gr.Gallery(label="Output Images", show_label=True, height=512, elem_id="output-image", columns=[2])
                save_output = gr.Checkbox(label="Save output image", value=True)
                info_box = gr.Markdown("", elem_id="output-info")
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
            out_imgs, info_md = process(prompt, raw, image, int(safety), use_seed, float(guidance_scale), int(num_images), output_format, output_aspect, image_size, aspect_ratio, float(image_prompt_strength), int(num_inference_steps), save_output, save_input)
            return out_imgs, info_md, gr.update(value=new_seed)
        # Run button click
        run_btn.click(run_all, inputs=[prompt, raw, image, safety, seed, lock_seed, guidance_scale, num_images, output_format, output_aspect, image_prompt_strength, num_inference_steps, save_output, save_input], outputs=[after, info_box, seed])
        # Prompt box: run on Ctrl+Enter
        prompt.submit(run_all, inputs=[prompt, raw, image, safety, seed, lock_seed, guidance_scale, num_images, output_format, output_aspect, image_prompt_strength, num_inference_steps, save_output, save_input], outputs=[after, info_box, seed], queue=True, preprocess=True)
        # Video upload: extract first frame and set as image input
        def handle_video_extract(video_path, rotate_degrees):
            if not video_path:
                return gr.update()
            try:
                img = extract_first_frame_from_video(video_path, rotate_degrees)
                return img
            except Exception as e:
                print(f"[ERROR] Could not extract frame: {e}")
                return gr.update()
        extract_btn.click(handle_video_extract, inputs=[video_upload, rotate_dropdown], outputs=[image])
    # Ensure .gitignore is updated and process active jobs on startup
    ensure_gitignore()
    process_active_jobs_on_startup()
    demo.launch(server_name=args.host, server_port=args.port, share=False)

if __name__ == "__main__":
    main() 
