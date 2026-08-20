"""CPU vision sidecar for vLLM --enable-mm-embeds.

Proxy: :8006 -> vLLM :18079. Rewrites image_url content parts into
image_embeds parts computed by the checkpoint's vision tower on CPU.

Env:
  SIDECAR_MODEL           local path or Hugging Face model ID
  SIDECAR_MODEL_REVISION  immutable Hugging Face revision
  SIDECAR_UPSTREAM   vLLM base URL (default http://127.0.0.1:18079)

Run:  python sidecar.py 8006
"""
import base64
import asyncio
import io
import json
import os
import sys

import httpx
import torch
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor

MODEL = os.environ.get(
    "SIDECAR_MODEL", "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"
)
MODEL_REVISION = os.environ.get(
    "SIDECAR_MODEL_REVISION", "69274a0d8dff5dd35bcee8290612f71e03b6e981"
)
UPSTREAM = os.environ.get("SIDECAR_UPSTREAM", "http://127.0.0.1:18079")
MAX_PIXELS = 1280 * 28 * 28  # cap ~1.28 MP like the Qwen3-VL default
MAX_IMAGE_BYTES = 25 * 1024 * 1024

app = FastAPI()


class VisionEncoder:
    def __init__(self) -> None:
        model_dir = MODEL if os.path.isdir(MODEL) else snapshot_download(
            repo_id=MODEL,
            revision=MODEL_REVISION,
            token=os.environ.get("HF_TOKEN") or None,
        )
        self.model_dir = model_dir
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        from transformers import AutoModel

        vision_cls = AutoModel._model_mapping[type(cfg.vision_config)]
        self.model = vision_cls(cfg.vision_config)
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as index_file:
            wm = json.load(index_file)["weight_map"]
        state = {}
        for shard in sorted(set(wm.values())):
            with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("model.visual."):
                        state[k[len("model.visual."):]] = f.get_tensor(k)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"vision load mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
        self.model.eval()
        print(f"[sidecar] vision tower loaded on CPU: {sum(v.numel() for v in self.model.parameters())} params")

    @torch.no_grad()
    def encode(self, image_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if img.width * img.height > MAX_PIXELS:
            scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        inputs = self.processor(images=[img], return_tensors="pt")
        out = self.model(
            hidden_states=inputs["pixel_values"],
            grid_thw=inputs["image_grid_thw"],
        )
        embeds = out.pooler_output.float().to(torch.bfloat16).contiguous()
        grid = inputs["image_grid_thw"][0].to(torch.int64).contiguous()  # [3]
        return embeds, grid


encoder: VisionEncoder | None = None
client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0))


def _load_image(url: str) -> bytes:
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        data = base64.b64decode(b64, validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds 25 MiB limit")
        return data
    if os.path.exists(url):
        if os.path.getsize(url) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds 25 MiB limit")
        with open(url, "rb") as image_file:
            return image_file.read()
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    data = response.content
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 25 MiB limit")
    return data


def _b64(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode()


async def _rewrite(messages: list) -> list:
    out = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part["image_url"]
                if isinstance(url, dict):
                    url = url.get("url", "")
                data = await asyncio.to_thread(_load_image, url)
                embeds, grid = await asyncio.to_thread(encoder.encode, data)
                parts.append({
                    "type": "image_embeds",
                    "image_embeds": {
                        "image_embeds": _b64(embeds),
                        "image_grid_thw": _b64(grid),
                    },
                })
            else:
                parts.append(part)
        new_msg = dict(msg)
        new_msg["content"] = parts
        out.append(new_msg)
    return out


@app.on_event("startup")
def startup() -> None:
    global encoder
    encoder = VisionEncoder()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "vision": encoder is not None,
        "model": MODEL,
        "revision": MODEL_REVISION,
    }


@app.post("/v1/chat/completions")
async def chat(req: Request) -> Response:
    body = await req.json()
    body["messages"] = await _rewrite(body.get("messages", []))
    stream = body.get("stream", False)
    upstream_req = client.build_request("POST", f"{UPSTREAM}/v1/chat/completions", json=body)
    if stream:
        upstream_resp = await client.send(upstream_req, stream=True)

        async def gen():
            async for line in upstream_resp.aiter_lines():
                yield line + "\n"

        return StreamingResponse(gen(), media_type="text/event-stream", status_code=upstream_resp.status_code)
    upstream_resp = await client.send(upstream_req)
    return Response(content=upstream_resp.content, status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get("content-type", "application/json"))


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8006
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
