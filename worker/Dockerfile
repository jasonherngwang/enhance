FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git curl gcc g++ && rm -rf /var/lib/apt/lists/*

ARG SEEDVR2_REF=v2.5.23
RUN git clone --depth 1 --branch ${SEEDVR2_REF} https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /app/SeedVR2
RUN pip install --no-cache-dir -r /app/SeedVR2/requirements.txt runpod==1.10.* boto3
RUN pip install --no-cache-dir sageattention==1.0.6

ARG DIT_REPO=numz/SeedVR2_comfyUI
ARG DIT_REF=main
ARG DIT_MODEL=seedvr2_ema_3b_fp8_e4m3fn.safetensors
RUN mkdir -p /models && \
    curl -fL --retry 8 --retry-delay 5 --retry-all-errors -C - -o /models/${DIT_MODEL} https://huggingface.co/${DIT_REPO}/resolve/${DIT_REF}/${DIT_MODEL} && \
    curl -fL --retry 8 --retry-delay 5 --retry-all-errors -C - -o /models/ema_vae_fp16.safetensors https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors

ENV CLI_DIR=/app/SeedVR2 MODEL_DIR=/models DIT_MODEL=${DIT_MODEL}
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
