"""SeedVR2 serverless worker (Runpod): upscale one video chunk per job.

job["input"]: chunk_url (http/s3), output_key, resolution (default 1080), batch_size (4n+1, default 9).
For S3 output, set BUCKET_ENDPOINT / BUCKET_ACCESS_KEY / BUCKET_SECRET_KEY / BUCKET_NAME
(otherwise the output stays in the container). Returns a stats dict, or {"error": ...}.
"""
import os, subprocess, time, uuid, urllib.request

import runpod
import torch

CLI_DIR = os.environ.get("CLI_DIR", "/app/SeedVR2")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
DIT_MODEL = os.environ.get("DIT_MODEL", "seedvr2_ema_3b_fp8_e4m3fn.safetensors")

def pick_attention():
    """Arch-aware backend: sageattn_2 is valid on Ampere/Ada/Hopper; sdpa is the safe fallback."""
    try:
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:  # Ampere+
            return "sageattn_2"
    except Exception:
        pass
    return "sdpa"

ATTENTION = pick_attention()

def verify_gpu():
    """Run a matmul and check it's finite: stop on a broken GPU at startup so Runpod
    reschedules the worker before it bills a job."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU verify failed: torch.cuda.is_available() is False")
    a = torch.randn(512, 512, device="cuda")
    b = a @ a.T
    torch.cuda.synchronize()
    if not torch.isfinite(b).all():
        raise RuntimeError("GPU verify failed: non-finite matmul result")
    print(f"GPU verify OK: {torch.cuda.get_device_name(0)}, attention={ATTENTION}", flush=True)

verify_gpu()

def s3():
    import boto3, re
    ep = os.environ["BUCKET_ENDPOINT"]
    # region must match the volume's datacenter (boto3's us-east-1 default is rejected); BUCKET_REGION overrides
    m = re.match(r"https://s3api-([a-z0-9-]+)\.runpod\.io", ep)
    region = os.environ.get("BUCKET_REGION") or (m.group(1) if m else None)
    return boto3.client("s3",
        endpoint_url=ep, region_name=region,
        aws_access_key_id=os.environ["BUCKET_ACCESS_KEY"],
        aws_secret_access_key=os.environ["BUCKET_SECRET_KEY"])

def fetch(url, dest):
    if url.startswith("s3://"):
        _, _, rest = url.partition("s3://")
        bucket, _, key = rest.partition("/")
        # stream get_object, not download_file (Runpod S3 rejects its HeadObject preflight)
        obj = s3().get_object(Bucket=bucket, Key=key)
        with open(dest, "wb") as f:
            for part in obj["Body"].iter_chunks(1 << 20):
                f.write(part)
    else:
        urllib.request.urlretrieve(url, dest)

def handler(job):
    ji = job.get("input") or {}
    chunk_url = ji.get("chunk_url")
    output_key = ji.get("output_key")
    if not chunk_url or not output_key:
        return {"error": "chunk_url and output_key are required"}
    resolution = int(ji.get("resolution", 1080))
    batch_size = int(ji.get("batch_size", 9))
    if batch_size % 4 != 1:
        return {"error": f"batch_size must be 4n+1, got {batch_size}"}

    work = f"/tmp/{uuid.uuid4().hex}"
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "in.mp4")
    dst = os.path.join(work, "out.mp4")
    try:
        fetch(chunk_url, src)
    except Exception as e:
        return {"error": f"fetch failed: {e}"}

    cmd = [
        "python", "inference_cli.py", src,
        "--output", dst, "--output_format", "mp4",
        "--model_dir", MODEL_DIR, "--dit_model", DIT_MODEL,
        "--resolution", str(resolution), "--batch_size", str(batch_size),
        "--temporal_overlap", "2", "--uniform_batch_size",
        "--color_correction", "lab", "--attention_mode", ATTENTION,
        "--video_backend", "ffmpeg",
        "--vae_encode_tiled", "--vae_decode_tiled",
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=CLI_DIR, capture_output=True, text=True, timeout=3600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(dst):
        return {"error": f"upscale failed rc={r.returncode}", "stderr_tail": r.stderr[-1500:]}

    if os.environ.get("BUCKET_ENDPOINT"):
        bucket = os.environ.get("BUCKET_NAME")
        if not bucket:
            return {"error": "BUCKET_ENDPOINT is set but BUCKET_NAME is not"}
        try:
            s3().upload_file(dst, bucket, output_key)
        except Exception as e:
            return {"error": f"upload failed: {e}"}
    else:
        output_key = f"LOCAL:{dst}"  # local mode (no S3 configured): leave output in container

    nframes = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", dst],
        capture_output=True, text=True).stdout.strip() or 0)
    return {"output_key": output_key, "frames": nframes, "seconds": round(dt, 1),
            "sec_per_frame": round(dt / max(1, nframes), 3), "attention_mode": ATTENTION}

runpod.serverless.start({"handler": handler})
