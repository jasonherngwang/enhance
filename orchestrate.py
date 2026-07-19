#!/usr/bin/env python3
"""Cloud orchestrator: upload chunks, fan out one /run job per chunk, poll status and
fleet /health, download outputs, record cost.

Env (./.env or environment): RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID (or --endpoint),
RUNPOD_S3_ENDPOINT, RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY, RUNPOD_VOLUME_ID.
Chunks travel via S3 storage, not the /run body (10MB cap). Times are in ms.
"""
import argparse, glob, json, os, sys, time, urllib.request

API = "https://api.runpod.ai/v2"
REST = "https://rest.runpod.io/v1"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

def http(method, url, body=None, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, method=method,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry {url.split('/')[-1]}: {e}", file=sys.stderr)
            time.sleep(2 ** i)

def s3c():
    import boto3, re
    from botocore.config import Config
    ep = os.environ["RUNPOD_S3_ENDPOINT"]
    # region must match the volume's datacenter (boto3 defaults to us-east-1, which is rejected)
    m = re.match(r"https://s3api-([a-z0-9-]+)\.runpod\.io", ep)
    cfg = Config(region_name=m.group(1) if m else None,
                 connect_timeout=30, read_timeout=300,
                 retries={"max_attempts": 6, "mode": "adaptive"})
    return boto3.client("s3", endpoint_url=ep, config=cfg,
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_KEY"])

def submit(ep, chunkdir, statedir, resolution, batch_size, http_base, exec_timeout_ms):
    man = json.load(open(os.path.join(chunkdir, "manifest.json")))
    os.makedirs(statedir, exist_ok=True)
    vol = os.environ.get("RUNPOD_VOLUME_ID")
    if not http_base and not vol:
        sys.exit("need RUNPOD_VOLUME_ID for S3 upload, or pass --http-base for pre-hosted chunks")
    prefix = os.path.basename(os.path.abspath(statedir))
    jobs = []
    for c in man["chunks"]:
        name = os.path.basename(c["file"])
        if http_base:
            chunk_url = f"{http_base.rstrip('/')}/{name}"
        else:
            key = f"{prefix}/in/{name}"
            print(f"upload {name} -> s3://{vol}/{key}")
            s3c().upload_file(c["file"], vol, key)
            chunk_url = f"s3://{vol}/{key}"
        body = {"input": {"chunk_url": chunk_url,
                          "output_key": f"{prefix}/out/up_{c['i']:04d}.mp4",
                          "resolution": resolution, "batch_size": batch_size}}
        if exec_timeout_ms:
            body["policy"] = {"executionTimeout": exec_timeout_ms}
        r = http("POST", f"{API}/{ep}/run", body)
        jobs.append({"chunk": c["i"], "job_id": r["id"], "chunk_url": chunk_url,
                     "submitted": time.time(), "status": r.get("status", "IN_QUEUE")})
        print(f"chunk {c['i']} -> job {r['id']}")
    state = {"endpoint": ep, "chunkdir": os.path.abspath(chunkdir), "prefix": prefix,
             "resolution": resolution, "batch_size": batch_size,
             "t_submit": time.time(), "jobs": jobs, "timeline": []}
    save(state, statedir)
    return state

def save(state, statedir):
    with open(os.path.join(statedir, "jobs.json"), "w") as f:
        json.dump(state, f, indent=1)

def poll(state, statedir, interval=10):
    ep = state["endpoint"]
    while True:
        health = http("GET", f"{API}/{ep}/health")
        for j in state["jobs"]:
            if j["status"] not in TERMINAL:
                s = http("GET", f"{API}/{ep}/status/{j['job_id']}")
                j.update({k: s[k] for k in
                          ("status", "delayTime", "executionTime", "workerId", "output")
                          if k in s})
        state["timeline"].append({"t": time.time(), "jobs": health.get("jobs"),
                                  "workers": health.get("workers")})
        save(state, statedir)
        counts = {}
        for j in state["jobs"]:
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        print(f"[{time.strftime('%H:%M:%S')}] {counts} workers={health.get('workers')}")
        if all(j["status"] in TERMINAL for j in state["jobs"]):
            return state
        time.sleep(interval)

def download(state, updir, tries=5):
    os.makedirs(updir, exist_ok=True)
    vol = os.environ["RUNPOD_VOLUME_ID"]
    for j in state["jobs"]:
        out = (j.get("output") or {})
        key = out.get("output_key", "")
        if j["status"] != "COMPLETED" or not key or key.startswith("LOCAL:"):
            continue
        dst = os.path.join(updir, f"up_{j['chunk']:04d}.mp4")
        for attempt in range(tries):
            try:
                obj = s3c().get_object(Bucket=vol, Key=key)
                tmp = dst + ".part"
                n = 0
                with open(tmp, "wb") as f:
                    for part in obj["Body"].iter_chunks(1 << 20):
                        f.write(part); n += len(part)
                if n == 0:
                    raise IOError("0 bytes streamed")
                os.replace(tmp, dst)
                print(f"download s3://{vol}/{key} -> {dst} ({n} B)")
                break
            except Exception as e:
                if attempt == tries - 1:
                    print(f"  DOWNLOAD FAILED {key}: {e}", file=sys.stderr)
                else:
                    time.sleep(2 ** attempt)

def receipt(state, statedir, rate_per_sec=None, billing=False):
    jobs = state["jobs"]
    done = [j for j in jobs if j["status"] == "COMPLETED"]
    exec_ms = sum(j.get("executionTime", 0) for j in jobs)
    delay_ms = sum(j.get("delayTime", 0) for j in jobs)
    frames = sum((j.get("output") or {}).get("frames", 0) for j in done)
    wall = (state["timeline"][-1]["t"] - state["t_submit"]) if state["timeline"] else 0
    rec = {"endpoint": state["endpoint"], "chunks": len(jobs), "completed": len(done),
           "failed": [j["job_id"] for j in jobs if j["status"] != "COMPLETED"],
           "frames": frames, "wall_clock_s": round(wall, 1),
           "exec_s_total": round(exec_ms / 1000, 1), "delay_s_total": round(delay_ms / 1000, 1),
           "sec_per_frame_gpu": round(exec_ms / 1000 / frames, 3) if frames else None,
           "workers_used": sorted({j.get("workerId") for j in jobs if j.get("workerId")}),
           "per_job": [{k: j.get(k) for k in
                        ("chunk", "job_id", "status", "delayTime", "executionTime", "workerId", "output")}
                       for j in jobs]}
    if rate_per_sec:
        rec["est_cost_exec_only"] = round(exec_ms / 1000 * rate_per_sec, 4)
        rec["est_cost_exec_plus_delay"] = round((exec_ms + delay_ms) / 1000 * rate_per_sec, 4)
        rec["rate_per_sec_used"] = rate_per_sec
    if billing:
        import datetime
        iso = lambda t: datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")
        t0, t1 = int(state["t_submit"]) - 86400, int(time.time()) + 86400
        try:
            b = http("GET", f"{REST}/billing/endpoints?startTime={iso(t0)}&endTime={iso(t1)}&bucketSize=hour")
            mine = [x for x in b if x.get("endpointId") == state["endpoint"]] if isinstance(b, list) else b
            rec["billing_raw"] = mine
            rec["billed_amount_usd"] = sum(x.get("amount", 0) for x in mine) if isinstance(mine, list) else None
            if isinstance(mine, list) and not mine:
                rec["billing_note"] = ("no billing rows yet for this endpoint — Runpod billing "
                                       "is daily-bucketed and lags; re-run `receipt --billing` later")
        except Exception as e:
            rec["billing_error"] = str(e)
    path = os.path.join(statedir, "receipt.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("per_job", "billing_raw")}, indent=1))
    print(f"receipt -> {path}")
    return rec

if __name__ == "__main__":
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    ap.add_argument("--statedir", default="runs/latest", help="state + receipt dir; its basename namespaces S3 keys")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="submit + poll + download + receipt")
    s = sub.add_parser("submit"); p = sub.add_parser("poll"); d = sub.add_parser("download")
    c = sub.add_parser("receipt")
    for x in (r, s):
        x.add_argument("--chunkdir", default="chunks")
        x.add_argument("--resolution", type=int, default=1080)
        x.add_argument("--batch-size", type=int, default=9)
        x.add_argument("--http-base", help="chunks pre-hosted here; skips S3 upload")
        x.add_argument("--exec-timeout-ms", type=int, default=3_600_000)
    for x in (r, d):
        x.add_argument("--updir", default="out_cloud")
    for x in (r, c):
        x.add_argument("--rate-per-sec", type=float, help="live GPU $/s for estimate")
        x.add_argument("--billing", action="store_true", help="reconcile vs REST /billing/endpoints")
    a = ap.parse_args()
    if a.cmd in ("run", "submit"):
        if not a.endpoint:
            sys.exit("need --endpoint or RUNPOD_ENDPOINT_ID")
        st = submit(a.endpoint, a.chunkdir, a.statedir, a.resolution, a.batch_size,
                    a.http_base, a.exec_timeout_ms)
        if a.cmd == "submit":
            sys.exit(0)
    else:
        st = json.load(open(os.path.join(a.statedir, "jobs.json")))
    if a.cmd in ("run", "poll"):
        st = poll(st, a.statedir)
    if a.cmd in ("run", "download"):
        if os.environ.get("RUNPOD_VOLUME_ID"):
            download(st, a.updir)
        else:
            print("no RUNPOD_VOLUME_ID -> skipping download (LOCAL-mode outputs stay in workers)")
    if a.cmd in ("run", "receipt"):
        receipt(st, a.statedir, a.rate_per_sec, a.billing)
