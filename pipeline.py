#!/usr/bin/env python3
"""Split a video into chunks, upscale each (local GPU or serverless fleet), stitch back.
Overlap-and-discard: each chunk starts OVERLAP frames early; stitch drops them.
"""
import argparse, json, os, subprocess, sys, time, math, glob, shlex

OVERLAP = 8

def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
load_env()

def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"FAIL: {cmd}\n{r.stderr[-2000:]}")
    return r.stdout

def q(s):
    return shlex.quote(str(s))

def probe(path):
    j = json.loads(sh(f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,pix_fmt,color_transfer -show_entries format=duration -of json {q(path)}"))
    st = j["streams"][0]
    num, den = map(int, st["r_frame_rate"].split("/"))
    fps = num / den
    dur = float(j["format"]["duration"])
    nb = int(st.get("nb_frames") or round(dur * fps))
    afr = st.get("avg_frame_rate", "0/0")
    an, ad = (int(x) for x in afr.split("/")) if "/" in afr and afr != "0/0" else (0, 1)
    avg = an / ad if ad else 0
    return {"fps": fps, "avg_fps": avg, "frames": nb, "dur": dur, "w": st["width"], "h": st["height"],
            "pix_fmt": st.get("pix_fmt", ""), "transfer": st.get("color_transfer", "")}

# HDR transfer characteristics that need tone-mapping to SDR (HLG, PQ)
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}

def normalize(src, outdir, fps, mode):
    """CFR/SDR/8-bit video-only intermediate for splitting. Phone footage is often VFR
    (audio drift) and HDR BT.2020 (washed out in an SDR pipeline). Returns (path, normalized)."""
    info = probe(src)
    is_hdr = info["transfer"] in HDR_TRANSFERS
    is_vfr = info["avg_fps"] and abs(info["fps"] - info["avg_fps"]) > 0.01
    is_10bit = "10" in info["pix_fmt"]
    if mode == "off" or (mode == "auto" and not (is_hdr or is_vfr or is_10bit)):
        return src, False
    norm = os.path.join(outdir, "_normalized.mp4")
    if is_hdr:
        # HLG/PQ BT.2020 -> linear light -> Hable tone-map -> BT.709 SDR, 8-bit.
        vf = ("zscale=t=linear:npl=100,format=gbrpf32le,tonemap=tonemap=hable:desat=0,"
              f"zscale=p=bt709:t=bt709:m=bt709:r=tv,format=yuv420p,fps={fps}")
    else:
        vf = f"fps={fps},format=yuv420p"
    print(f"normalize: hdr={is_hdr} vfr={is_vfr} 10bit={is_10bit} -> CFR {fps}fps SDR 8-bit intermediate")
    sh(f"ffmpeg -y -loglevel error -i {q(src)} -map 0:v:0 -vf {q(vf)} -vsync cfr -r {fps} "
       f"-c:v libx264 -crf 12 -pix_fmt yuv420p {q(norm)}")
    return norm, True

def split(src, chunk_sec, outdir, fps=30, normalize_mode="auto"):
    os.makedirs(outdir, exist_ok=True)
    audio_src = src                                  # audio always remuxed from the ORIGINAL
    src, was_norm = normalize(src, outdir, fps, normalize_mode)
    info = probe(src)
    fps = info["fps"]; total = info["frames"]
    cf = max(1, round(chunk_sec * fps))          # frames per chunk
    n = math.ceil(total / cf)
    manifest = {"src": os.path.abspath(src), "audio_src": os.path.abspath(audio_src),
                "normalized": was_norm, "fps": fps, "total_frames": total,
                "chunk_frames": cf, "overlap": OVERLAP, "chunks": []}
    for i in range(n):
        start = max(0, i * cf - (OVERLAP if i > 0 else 0))
        length = (i * cf + cf) - start
        out = os.path.join(outdir, f"chunk_{i:04d}.mp4")
        vf = f"select=between(n\\,{start}\\,{start+length-1}),setpts=PTS-STARTPTS"
        sh(f"ffmpeg -y -loglevel error -i {q(src)} -vf {q(vf)} -vsync 0 -an -c:v libx264 -crf 10 -pix_fmt yuv420p {q(out)}")
        manifest["chunks"].append({"i": i, "file": out, "start": start,
                                   "frames": min(length, total - start),
                                   "discard_head": OVERLAP if i > 0 else 0})
    sh(f"ffmpeg -y -loglevel error -i {q(audio_src)} -vn -c:a aac -b:a 192k {q(os.path.join(outdir, 'audio.m4a'))} || true")
    with open(os.path.join(outdir, "manifest.json"), "w") as f: json.dump(manifest, f, indent=1)
    print(f"split: {total} frames @ {fps:.2f}fps -> {n} chunks of ~{cf}f (+{OVERLAP}f overlap), manifest written")

def upscale_local(chunkdir, outdir, resolution, cli, model_dir, batch_size=5):
    os.makedirs(outdir, exist_ok=True)
    man = json.load(open(os.path.join(chunkdir, "manifest.json")))
    t0 = time.time(); report = []
    for c in man["chunks"]:
        out = os.path.join(outdir, f"up_{c['i']:04d}.mp4")
        if os.path.exists(out): print(f"skip {out}"); continue
        t1 = time.time()
        py = os.environ.get("SEEDVR2_PYTHON", "python3")
        sh(f"cd {q(os.path.dirname(cli))} && {q(py)} {q(os.path.basename(cli))} "
           f"{q(os.path.abspath(c['file']))} --output {q(os.path.abspath(out))} --output_format mp4 "
           f"--model_dir {q(model_dir)} --dit_model seedvr2_ema_3b_fp8_e4m3fn.safetensors "
           f"--resolution {resolution} --batch_size {batch_size} --temporal_overlap 2 --uniform_batch_size --color_correction lab "
           f"--attention_mode sageattn_2 --video_backend ffmpeg --vae_encode_tiled --vae_decode_tiled")
        dt = time.time() - t1
        report.append({"chunk": c['i'], "frames": c['frames'], "sec": round(dt,1), "sec_per_frame": round(dt/max(1,c['frames']),2)})
        print(f"chunk {c['i']}: {dt:.0f}s ({dt/max(1,c['frames']):.2f}s/f)")
    json.dump(report, open(os.path.join(outdir, "timing.json"), "w"), indent=1)
    print(f"upscale-local done in {time.time()-t0:.0f}s total")

def stitch(chunkdir, updir, out):
    man = json.load(open(os.path.join(chunkdir, "manifest.json")))
    fps = man["fps"]; parts = []
    for c in man["chunks"]:
        up = os.path.join(updir, f"up_{c['i']:04d}.mp4")
        trimmed = os.path.join(updir, f"trim_{c['i']:04d}.mp4")
        d = c["discard_head"]
        vf = f"select=gte(n\\,{d}),setpts=PTS-STARTPTS"
        sh(f"ffmpeg -y -loglevel error -i {q(up)} -vf {q(vf)} -vsync 0 -an -c:v libx264 -crf 14 -pix_fmt yuv420p {q(trimmed)}")
        parts.append(trimmed)
    lst = os.path.join(updir, "concat.txt")
    with open(lst, "w") as f:
        for p in parts: f.write(f"file \x27{os.path.abspath(p)}\x27\n")
    audio = os.path.join(chunkdir, "audio.m4a")
    amap = f"-i {q(audio)} -map 0:v -map 1:a -c:a copy -shortest" if os.path.exists(audio) and os.path.getsize(audio)>1000 else ""
    sh(f"ffmpeg -y -loglevel error -f concat -safe 0 -i {q(lst)} {amap} -c:v copy -r {fps} {q(out)}")
    print(f"stitched -> {out}")
    print(sh(f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 {q(out)}").strip())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("src"); s.add_argument("--chunk-sec", type=float, default=10); s.add_argument("--outdir", default="chunks")
    s.add_argument("--fps", type=float, default=30, help="target CFR when normalizing")
    s.add_argument("--normalize", choices=["auto", "force", "off"], default="auto", help="CFR/SDR/8-bit pre-pass; auto-detects HDR/VFR/10-bit")
    u = sub.add_parser("upscale-local"); u.add_argument("--chunkdir", default="chunks"); u.add_argument("--outdir", default="out"); u.add_argument("--resolution", type=int, default=1080); u.add_argument("--batch-size", type=int, default=5)
    u.add_argument("--cli", default=os.environ.get("SEEDVR2_CLI"), help="path to SeedVR2 inference_cli.py (or set SEEDVR2_CLI)")
    u.add_argument("--model-dir", default=os.environ.get("SEEDVR2_MODEL_DIR"), help="dir with SeedVR2 weights (or set SEEDVR2_MODEL_DIR)")
    t = sub.add_parser("stitch"); t.add_argument("--chunkdir", default="chunks"); t.add_argument("--updir", default="out"); t.add_argument("--out", default="upscaled.mp4")
    a = ap.parse_args()
    if a.cmd == "split": split(a.src, a.chunk_sec, a.outdir, a.fps, a.normalize)
    elif a.cmd == "upscale-local":
        if not a.cli or not a.model_dir:
            sys.exit("need --cli and --model-dir (or SEEDVR2_CLI / SEEDVR2_MODEL_DIR in env or ./.env)")
        upscale_local(a.chunkdir, a.outdir, a.resolution, a.cli, a.model_dir, a.batch_size)
    elif a.cmd == "stitch": stitch(a.chunkdir, a.updir, a.out)
