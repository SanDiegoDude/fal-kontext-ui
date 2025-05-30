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

def log_activity(prompt, input_url, output_url):
    # Roll log if needed
    if os.path.exists(ACTIVITY_LOG_FILE) and os.path.getsize(ACTIVITY_LOG_FILE) > MAX_LOG_SIZE:
        shutil.move(ACTIVITY_LOG_FILE, OLD_ACTIVITY_LOG_FILE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ACTIVITY_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, prompt, input_url, output_url])

def process_active_jobs_on_startup():
    jobs = list_active_jobs()
    for request_id, ts, prompt, input_url in jobs:
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
                            log_activity(prompt, input_url, output_url)
                # Save images to output dir
                if images_out and urls_out:
                    outdir = os.path.join("output", datetime.now().strftime("%Y%m%d"))
                    os.makedirs(outdir, exist_ok=True)
                    for img, out_url in zip(images_out, urls_out):
                        fal_hash = extract_fal_hash(out_url)
                        if fal_hash:
                            ext = os.path.splitext(out_url)[1]
                            filename = os.path.join(outdir, f"{fal_hash}{ext}")
                            img.save(filename)
                            debug(f"Saved output image: {filename}")
                remove_active_job(request_id)
        except Exception as e:
            debug(f"Error resuming job {request_id}: {e}")

def call_kontext(prompt, image_url, safety, seed, guidance_scale, num_images, output_format, raw, image_prompt_strength, num_inference_steps):
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
    job_id_holder = {}
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
                    # Log activity
                    log_activity(prompt, image_url, output_url)
    # Remove from .active if job_id is known
    if 'id' in job_id_holder:
        remove_active_job(job_id_holder['id'])
    return images_out, urls_out

def process(prompt, raw, image, safety, seed, guidance_scale, num_images, output_format, image_prompt_strength, num_inference_steps, save_output):
    if isinstance(image, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format='PNG')
            tmp_path = tmp.name
        try:
            debug(f"Uploading input image to ImgBB: {tmp_path}")
            url = upload_image_to_imgbb(tmp_path)
        finally:
            os.remove(tmp_path)
    elif isinstance(image, str) and image.startswith("http"):
        url = image
    else:
        return None, ""
    out_imgs, out_urls = call_kontext(prompt, url, safety, seed, guidance_scale, num_images, output_format, raw, image_prompt_strength, num_inference_steps)
    # Always save images if requested, regardless of how they were produced
    if save_output and out_imgs and out_urls:
        outdir = os.path.join("output", datetime.now().strftime("%Y%m%d"))
        os.makedirs(outdir, exist_ok=True)
        for img, out_url in zip(out_imgs, out_urls):
            fal_hash = extract_fal_hash(out_url)
            if fal_hash:
                ext = os.path.splitext(out_url)[1]
                filename = os.path.join(outdir, f"{fal_hash}{ext}")
                debug(f"Saving output image to: {filename}")
                img.save(filename)
                debug(f"Saved output image: {filename}")
                print(f"Saved output image: {filename}")
    # Print URLs to CLI
    for out_url in out_urls:
        print(f"Output URL: {out_url}")
    # Only show the first image in the UI, but info box lists all URLs
    info_md = f"**Prompt:** {prompt}\n\n**Seed:** {seed}\n\n**Output URL(s):**\n" + "\n".join(out_urls if out_urls else ['N/A'])
    return (out_imgs[0] if out_imgs else None), info_md

def extract_fal_hash(url):
    # Extract everything after /koala/ or /files/koala/ in the URL
    m = re.search(r"(?:/koala/|/files/koala/)([^/?#]+)", url)
    if m:
        return m.group(1)
    return None

def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="Kontext UI - Open Source Fal Kontext Demo")
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
                    safety = gr.Slider(label="Safety Tolerance (1=Strict, 6=Permissive)", minimum=1, maximum=6, step=1, value=5)
            with gr.Column():
                image = gr.Image(label="Input Image", type="pil", height=512, show_label=True, elem_id="input-image")
                after = gr.Image(label="Output Image", show_label=True, height=512, elem_id="output-image")
                save_output = gr.Checkbox(label="Save output image", value=True)
                info_box = gr.Markdown("", elem_id="output-info")
        last_seed = {"value": default_seed}
        def run_all(prompt, raw, image, safety, seed, lock_seed, guidance_scale, num_images, output_format, image_prompt_strength, num_inference_steps, save_output):
            if lock_seed:
                use_seed = int(seed)
            else:
                use_seed = random.randint(0, 2**32 - 1)
                last_seed["value"] = use_seed
            return process(prompt, raw, image, int(safety), use_seed, float(guidance_scale), int(num_images), output_format, float(image_prompt_strength), int(num_inference_steps), save_output)
        # Run button click
        run_btn.click(run_all, inputs=[prompt, raw, image, safety, seed, lock_seed, guidance_scale, num_images, output_format, image_prompt_strength, num_inference_steps, save_output], outputs=[after, info_box])
        # Prompt box: run on Ctrl+Enter
        prompt.submit(run_all, inputs=[prompt, raw, image, safety, seed, lock_seed, guidance_scale, num_images, output_format, image_prompt_strength, num_inference_steps, save_output], outputs=[after, info_box], queue=True, preprocess=True)

    # Ensure .gitignore is updated and process active jobs on startup
    ensure_gitignore()
    process_active_jobs_on_startup()

    demo.launch(server_name=args.host, server_port=args.port, share=False)

if __name__ == "__main__":
    main() 