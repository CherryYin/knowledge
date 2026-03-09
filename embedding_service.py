"""
Embedding 服务模块

两种部署模式:
1. QwenMultiModalEmbedding  — 本地加载 Qwen2.5-VL-Embedding（Transformers）
2. VLLMEmbeddingClient      — 调用 vLLM 部署的 OpenAI 兼容 Embedding API（生产推荐）

两种模式均支持:
- 纯文本 Embedding
- 纯图片 Embedding
- 文本 + 图片多模态融合 Embedding

向量空间统一，维度默认 1536。
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────
#  本地 Qwen 多模态 Embedding（Transformers）
# ─────────────────────────────────────────────

class QwenMultiModalEmbedding:
    """
    基于 Qwen2.5-VL-Embedding 的本地多模态 Embedding 服务

    三种输入模式 → 同一向量空间:
      1. 纯文本  → 文本向量
      2. 纯图片  → 图片向量
      3. 文本+图片 → 融合向量

    安装依赖: pip install transformers torch Pillow
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-Embedding",
        device: str = "auto",
        max_text_length: int = 8192,
        batch_size: int = 32,
        dimension: int = 1536,
    ):
        self.model_name = model_name
        self.device = device
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.dimension = dimension
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. "
                "Run: pip install transformers torch"
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def processor(self):
        self._load_model()
        return self._processor

    # ── 公开接口 ──────────────────────────────────────────────

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """批量文本 Embedding，返回 (N, dim) numpy 数组"""
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            messages = [
                [{"role": "user", "content": [{"type": "text", "text": t}]}]
                for t in batch
            ]
            all_embeddings.append(self._encode_batch(messages))
        return np.vstack(all_embeddings)

    def embed_image(self, image_paths: list[str]) -> np.ndarray:
        """批量图片 Embedding，返回 (N, dim) numpy 数组"""
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i : i + self.batch_size]
            messages = [
                [{"role": "user", "content": [{"type": "image", "image": p}]}]
                for p in batch
            ]
            all_embeddings.append(self._encode_batch(messages))
        return np.vstack(all_embeddings)

    def embed_multimodal(
        self,
        texts: list[str],
        image_paths: list[Optional[str]],
    ) -> np.ndarray:
        """文本 + 图片融合 Embedding，返回 (N, dim) numpy 数组"""
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_images = image_paths[i : i + self.batch_size]
            messages = []
            for text, img_path in zip(batch_texts, batch_images):
                content: list[dict] = []
                if img_path and Path(img_path).exists():
                    content.append({"type": "image", "image": img_path})
                content.append({"type": "text", "text": text})
                messages.append([{"role": "user", "content": content}])
            all_embeddings.append(self._encode_batch(messages))
        return np.vstack(all_embeddings)

    # ── 内部编码 ──────────────────────────────────────────────

    def _encode_batch(self, messages_batch: list) -> np.ndarray:
        import torch
        from PIL import Image

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages_batch
        ]
        image_inputs: list = []
        for msg in messages_batch:
            for m in msg:
                for item in m.get("content", []):
                    if item.get("type") == "image":
                        image_inputs.append(Image.open(item["image"]).convert("RGB"))

        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = self._mean_pool(
                outputs.hidden_states[-1], inputs["attention_mask"]
            )

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().numpy()

    @staticmethod
    def _mean_pool(hidden_states, attention_mask) -> "torch.Tensor":
        import torch

        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_emb = torch.sum(hidden_states * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask


# ─────────────────────────────────────────────
#  vLLM Embedding 客户端（生产推荐）
# ─────────────────────────────────────────────

class VLLMEmbeddingClient:
    """
    调用 vLLM 部署的 OpenAI 兼容 Embedding API

    vLLM 部署示例:
        python -m vllm.entrypoints.openai.api_server \\
            --model Qwen/Qwen2.5-VL-Embedding \\
            --task embedding \\
            --dtype bfloat16 \\
            --max-model-len 8192 \\
            --port 8100

    安装依赖: pip install httpx
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8100",
        model: str = "Qwen/Qwen2.5-VL-Embedding",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ── 异步接口 ──────────────────────────────────────────────

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        """批量文本 Embedding（异步）"""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [d["embedding"] for d in data["data"]]

    async def embed_multimodal(
        self,
        text: str,
        image_path: Optional[str] = None,
    ) -> list[float]:
        """单条文本 + 图片多模态 Embedding（异步）"""
        import httpx

        content: list[dict] = []
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = Path(image_path).suffix.lstrip(".") or "png"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{b64}"},
            })
        content.append({"type": "text", "text": text})

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "model": self.model,
                    "input": [{"type": "multimodal", "content": content}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]

    # ── 同步包装 ──────────────────────────────────────────────

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        """embed_text 的同步包装"""
        return asyncio.run(self.embed_text(texts))

    def embed_multimodal_sync(
        self, text: str, image_path: Optional[str] = None
    ) -> list[float]:
        """embed_multimodal 的同步包装"""
        return asyncio.run(self.embed_multimodal(text, image_path))


# ─────────────────────────────────────────────
#  工厂函数
# ─────────────────────────────────────────────

def create_embedding_service(
    use_vllm: bool = True,
    vllm_base_url: Optional[str] = None,
    vllm_model: Optional[str] = None,
    local_model_name: Optional[str] = None,
    device: str = "auto",
) -> "VLLMEmbeddingClient | QwenMultiModalEmbedding":
    """
    创建 Embedding 服务实例

    优先使用 vLLM 客户端（生产），可通过 use_vllm=False 切换到本地模型。

    环境变量支持:
        EMBEDDING_BASE_URL  : vLLM 服务地址（默认 http://localhost:8100）
        EMBEDDING_MODEL     : 模型名称
        USE_LOCAL_EMBEDDING : 设为 "true" 时使用本地 Transformers 模型
    """
    if os.getenv("USE_LOCAL_EMBEDDING", "").lower() in ("1", "true", "yes"):
        use_vllm = False

    if use_vllm:
        return VLLMEmbeddingClient(
            base_url=vllm_base_url or os.getenv("EMBEDDING_BASE_URL", "http://localhost:8100"),
            model=vllm_model or os.getenv("EMBEDDING_MODEL", "Qwen/Qwen2.5-VL-Embedding"),
        )
    else:
        return QwenMultiModalEmbedding(
            model_name=local_model_name or os.getenv(
                "EMBEDDING_MODEL", "Qwen/Qwen2.5-VL-Embedding"
            ),
            device=device,
        )
