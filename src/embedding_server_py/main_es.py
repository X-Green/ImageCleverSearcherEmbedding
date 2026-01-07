from abc import ABC, abstractmethod
import base64
import io
import os
import time
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from result import Result, Err, OK, is_ok, is_err
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

try:
    from optimum.onnxruntime import (
        ORTModelForFeatureExtraction,
        ORTModelForZeroShotImageClassification,
    )
    _OPTIMUM_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # pragma: no cover
    # Allow TorchInferenceEngine to work even if Optimum/onnxruntime isn't installed.
    ORTModelForFeatureExtraction = None  # type: ignore[assignment]
    ORTModelForZeroShotImageClassification = None  # type: ignore[assignment]
    _OPTIMUM_IMPORT_ERROR = str(e)

from transformers import AutoModel, AutoModelForZeroShotImageClassification, AutoProcessor

# --- Configuration ---
# Define where your ONNX models are stored (directories)
MODEL_PATHS = {
    "siglip2": "models/siglip2",
    "dinov2": "models/dinov2"
}

# Security: only allow file:/// inputs if explicitly enabled.
ALLOW_LOCAL_FILE_URIS = os.getenv(
    "EMBEDDING_SERVER_ALLOW_LOCAL_FILE_URIS", "0") == "1"

# Networking: bound downloaded image size.
MAX_IMAGE_BYTES = int(
    os.getenv("EMBEDDING_SERVER_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))

# Uvicorn defaults can hang on slow URLs; keep bounded.
URL_TIMEOUT_SECONDS = float(
    os.getenv("EMBEDDING_SERVER_URL_TIMEOUT_SECONDS", "10"))

# --- Pydantic Models ---


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]
                 ] = Field(..., description="The input text or base64 image to embed.")
    model: str = Field(..., description="The ID of the model to use.")
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"
    permission: List[Any] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: List[str]

# --- Model Inference Engine ---


class BaseInferenceEngine(ABC):
    @abstractmethod
    def load_models(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def loaded_model_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def ensure_model_available(self, model_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def call_processor(
        self,
        model_name: str,
        *,
        text: Optional[List[str]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        return_tensors: str = "pt",
        padding: Optional[bool] = None,
        truncation: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Call the model-specific processor (tokenizer / image processor)."""
        raise NotImplementedError

    @abstractmethod
    def call_model(
        self,
        model_name: str,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """Call the underlying model forward pass."""
        raise NotImplementedError


class ONNXInferenceEngine(BaseInferenceEngine):
    def __init__(self):
        print("ONNX Inference Engine Initializing...")
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.model_load_errors: Dict[str, str] = {}
        self.created_ts = int(time.time())

        self.provider = "CPUExecutionProvider"
        self.providers: List[str] = ["CPUExecutionProvider"]
        self.session_options = None

        try:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            print(f"Available ONNX providers: {available_providers}")

            # Provider priority: OpenVINO > CUDA > DirectML > CPU
            desired = ["OpenVINOExecutionProvider", "CUDAExecutionProvider",
                       "DmlExecutionProvider", "CPUExecutionProvider"]
            self.providers = [p for p in desired if p in available_providers]
            self.provider = self.providers[0] if self.providers else "CPUExecutionProvider"

            if "OpenVINOExecutionProvider" not in available_providers:
                print(
                    "Tip: Install onnxruntime-openvino for faster inference on Intel devices.")

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session_options = session_options
        except Exception as e:
            print(f"Warning: failed to configure onnxruntime ({e}).")

    def load_models(self):
        """Loads ONNX models into memory."""
        if ORTModelForFeatureExtraction is None or ORTModelForZeroShotImageClassification is None:
            msg = (
                "Optimum ONNXRuntime is not available. "
                + (f"Import error: {_OPTIMUM_IMPORT_ERROR}" if _OPTIMUM_IMPORT_ERROR else "")
            )
            for model_name in MODEL_PATHS.keys():
                self.model_load_errors[model_name] = msg
            print(msg)
            return

        for model_name, model_path in MODEL_PATHS.items():
            model_dir = Path(model_path)
            if not model_dir.exists():
                self.model_load_errors[model_name] = f"Model path not found: {model_path}"
                print(self.model_load_errors[model_name])
                continue

            print(
                f"Loading model {model_name} from {model_path} with {self.providers}...")
            try:
                if model_name == "siglip2":
                    self.models[model_name] = ORTModelForZeroShotImageClassification.from_pretrained(
                        model_path,
                        provider=self.provider,
                        session_options=self.session_options,
                    )
                    self.processors[model_name] = AutoProcessor.from_pretrained(
                        model_path)
                elif model_name == "dinov2":
                    self.models[model_name] = ORTModelForFeatureExtraction.from_pretrained(
                        model_path,
                        provider=self.provider,
                        session_options=self.session_options,
                    )
                    self.processors[model_name] = AutoProcessor.from_pretrained(
                        model_path)
                else:
                    raise RuntimeError(f"Unknown model key: {model_name}")
                print(f"Model {model_name} loaded successfully.")
            except Exception as e:
                msg = f"Failed to load {model_name}: {e}"
                self.model_load_errors[model_name] = msg
                print(msg)

    def loaded_model_names(self) -> List[str]:
        return sorted(self.models.keys())

    def ensure_model_available(self, model_name: str) -> None:
        if model_name not in MODEL_PATHS:
            raise HTTPException(
                status_code=400, detail=f"Model {model_name} not supported. Available: {list(MODEL_PATHS.keys())}")
        if not Path(MODEL_PATHS[model_name]).exists():
            raise HTTPException(
                status_code=404, detail=f"Model path not found for {model_name}")
        if model_name not in self.models:
            err = self.model_load_errors.get(model_name) or "Model not loaded"
            raise HTTPException(status_code=500, detail=err)

    def call_processor(
        self,
        model_name: str,
        *,
        text: Optional[List[str]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        return_tensors: str = "pt",
        padding: Optional[bool] = None,
        truncation: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        self.ensure_model_available(model_name)
        processor = self.processors.get(model_name)
        if processor is None:
            raise HTTPException(
                status_code=500, detail=f"Processor not loaded for {model_name}")
        effective_kwargs: Dict[str, Any] = dict(kwargs)
        if text is not None:
            effective_kwargs["text"] = text
        if images is not None:
            effective_kwargs["images"] = images
        if return_tensors is not None:
            effective_kwargs["return_tensors"] = return_tensors
        if padding is not None:
            effective_kwargs["padding"] = padding
        if truncation is not None:
            effective_kwargs["truncation"] = truncation
        return processor(**effective_kwargs)

    def call_model(
        self,
        model_name: str,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        self.ensure_model_available(model_name)
        model = self.models.get(model_name)
        if model is None:
            raise HTTPException(
                status_code=500, detail=f"Model not loaded for {model_name}")
        effective_kwargs: Dict[str, Any] = dict(kwargs)
        if input_ids is not None:
            effective_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            effective_kwargs["attention_mask"] = attention_mask
        if pixel_values is not None:
            effective_kwargs["pixel_values"] = pixel_values
        return model(**effective_kwargs)


class TorchInferenceEngine(BaseInferenceEngine):
    def __init__(self):
        print("Torch Inference Engine Initializing...")
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.model_load_errors: Dict[str, str] = {}
        self.created_ts = int(time.time())

        # Mirrors ONNX engine's provider selection concept.
        # Values: "auto" (default), "cpu", "cuda"
        device_pref = os.getenv(
            "EMBEDDING_SERVER_TORCH_DEVICE", "auto").strip().lower()
        if device_pref == "cuda":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        elif device_pref == "xpu":
            self.device = torch.device(
                "xpu" if torch.xpu.is_available() else "cpu")
        elif device_pref == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available(
            ) else "xpu" if torch.xpu.is_available() else "cpu")
        print("Using Torch device", self.device)

        # Model IDs mirror setup_models.py
        self.model_ids: Dict[str, str] = {
            "siglip2": os.getenv(
                "EMBEDDING_SERVER_TORCH_SIGLIP2_ID",
                "google/siglip2-base-patch16-512",
            ),
            "dinov2": os.getenv(
                "EMBEDDING_SERVER_TORCH_DINOV2_ID",
                "facebook/dinov2-base",
            ),
        }

        # Optional: pin cache directory for HF downloads.
        self.cache_dir: Optional[str] = os.getenv(
            "EMBEDDING_SERVER_TORCH_CACHE_DIR")

    @staticmethod
    def _to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: TorchInferenceEngine._to_device(v, device) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [TorchInferenceEngine._to_device(
                v, device) for v in value]
            return tuple(converted) if isinstance(value, tuple) else converted
        return value

    def load_models(self) -> None:
        """Loads PyTorch/Transformers models into memory."""
        for model_name in MODEL_PATHS.keys():
            model_id = self.model_ids.get(model_name)
            if not model_id:
                self.model_load_errors[model_name] = "Missing model id"
                print(self.model_load_errors[model_name])
                continue

            print(
                f"Loading Torch model {model_name} from {model_id} on {self.device}...")
            try:
                processor = AutoProcessor.from_pretrained(
                    model_id, cache_dir=self.cache_dir)

                if model_name == "siglip2":
                    # Use the retrieval/contrastive head so `get_text_features` and
                    # `get_image_features` map to the same embedding space.
                    model = AutoModelForZeroShotImageClassification.from_pretrained(
                        model_id,
                        cache_dir=self.cache_dir,
                    )
                else:
                    model = AutoModel.from_pretrained(
                        model_id, cache_dir=self.cache_dir)
                model.eval()
                model.to(self.device)

                self.processors[model_name] = processor
                self.models[model_name] = model
                print(f"Model {model_name} loaded successfully.")
            except Exception as e:
                msg = f"Failed to load {model_name}: {e}"
                self.model_load_errors[model_name] = msg
                print(msg)

    def loaded_model_names(self) -> List[str]:
        return sorted(self.models.keys())

    def ensure_model_available(self, model_name: str) -> None:
        # Keep same contract as ONNXInferenceEngine (400 unsupported, 500 not loaded).
        if model_name not in MODEL_PATHS:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not supported. Available: {list(MODEL_PATHS.keys())}",
            )
        if model_name not in self.models:
            err = self.model_load_errors.get(model_name) or "Model not loaded"
            raise HTTPException(status_code=500, detail=err)

    def call_processor(
        self,
        model_name: str,
        *,
        text: Optional[List[str]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        return_tensors: str = "pt",
        padding: Optional[bool] = None,
        truncation: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        self.ensure_model_available(model_name)
        processor = self.processors.get(model_name)
        if processor is None:
            raise HTTPException(
                status_code=500, detail=f"Processor not loaded for {model_name}")
        effective_kwargs: Dict[str, Any] = dict(kwargs)
        if text is not None:
            effective_kwargs["text"] = text
        if images is not None:
            effective_kwargs["images"] = images
        if return_tensors is not None:
            effective_kwargs["return_tensors"] = return_tensors
        if padding is not None:
            effective_kwargs["padding"] = padding
        if truncation is not None:
            effective_kwargs["truncation"] = truncation
        return processor(**effective_kwargs)

    def call_model(
        self,
        model_name: str,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        self.ensure_model_available(model_name)
        model = self.models.get(model_name)
        if model is None:
            raise HTTPException(
                status_code=500, detail=f"Model not loaded for {model_name}")

        effective_kwargs: Dict[str, Any] = dict(kwargs)
        if input_ids is not None:
            effective_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            effective_kwargs["attention_mask"] = attention_mask
        if pixel_values is not None:
            effective_kwargs["pixel_values"] = pixel_values

        moved_kwargs = self._to_device(effective_kwargs, self.device)

        with torch.inference_mode():
            # SigLIP2: prefer the dedicated feature-extraction APIs. This avoids
            # requiring both modalities (no dummy images/text) and produces
            # embeddings intended for cross-modal similarity.
            if model_name == "siglip2":
                from types import SimpleNamespace

                text_embeds = None
                image_embeds = None

                if hasattr(model, "get_text_features") and moved_kwargs.get("input_ids") is not None:
                    text_embeds = model.get_text_features(
                        input_ids=moved_kwargs.get("input_ids"),
                        attention_mask=moved_kwargs.get("attention_mask"),
                    )

                if hasattr(model, "get_image_features") and moved_kwargs.get("pixel_values") is not None:
                    image_embeds = model.get_image_features(
                        pixel_values=moved_kwargs.get("pixel_values"),
                    )

                if text_embeds is not None or image_embeds is not None:
                    return SimpleNamespace(text_embeds=text_embeds, image_embeds=image_embeds)

                # Fallback: try a normal forward pass and extract projected embeds.
                outputs = model(**moved_kwargs)
                if hasattr(outputs, "text_embeds") or hasattr(outputs, "image_embeds"):
                    return outputs

                raise HTTPException(
                    status_code=500,
                    detail="SigLIP2 model output missing text/image embeddings",
                )

            return model(**moved_kwargs)

    def process_and_forward(
            self, model_name: str, text: Optional[list[str]], images: Optional[list[Image.Image]]
    ) -> Result[tuple(Optional[list], Optional[list]), str]:
        if model_name == "siglip2":
            if not text and not images:
                return Err("No text or image given to siglip2")
            text_embd_result = None
            img_embd_result = None
            if text:
                pass
            
            if 


def _is_data_uri_image(s: str) -> bool:
    s2 = s.strip()
    return s2.startswith("data:image/") and ";base64," in s2[:80]


def _is_http_url(s: str) -> bool:
    s2 = s.strip().lower()
    return s2.startswith("http://") or s2.startswith("https://")


def _is_file_uri(s: str) -> bool:
    return s.strip().lower().startswith("file:///")


def _decode_data_uri_image(data_uri: str) -> Image.Image:
    # data:image/png;base64,AAAA
    header, b64 = data_uri.split(",", 1)
    if ";base64" not in header:
        raise ValueError("data URI is not base64")
    raw = base64.b64decode(b64, validate=True)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image too large")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _read_url_bytes(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "embedding-server/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SECONDS) as resp:
        # Best-effort cap by content-length; still cap during read.
        cl = resp.headers.get("Content-Length")
        if cl is not None:
            try:
                cl_int = int(cl)
            except (ValueError, TypeError):
                # ignore parse errors
                pass
            else:
                if cl_int > MAX_IMAGE_BYTES:
                    raise ValueError("image too large")
        data = resp.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image too large")
        return data


def _decode_url_image(url: str) -> Image.Image:
    data = _read_url_bytes(url)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _decode_file_uri_image(file_uri: str) -> Image.Image:
    if not ALLOW_LOCAL_FILE_URIS:
        raise PermissionError("Local file URIs are disabled")
    parsed = urllib.parse.urlparse(file_uri)
    if parsed.scheme.lower() != "file":
        raise ValueError("not a file URI")

    # file:///C:/path (Windows) OR file:///tmp/x.png (POSIX)
    # file://server/share/x.png (UNC / network share)
    if parsed.netloc and parsed.netloc.lower() not in ("", "localhost"):
        raw_path = f"//{parsed.netloc}{parsed.path}"
    else:
        raw_path = parsed.path

    path_str = urllib.request.url2pathname(raw_path)
    file_path = Path(path_str)
    if not file_path.exists():
        raise FileNotFoundError("Local file not found")
    if file_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("image too large")
    return Image.open(str(file_path)).convert("RGB")


def _try_load_image_from_input(s: str) -> Tuple[Optional[Image.Image], str]:
    """Returns (image_or_none, kind) where kind in {base64,url,file,text}."""
    s2 = s.strip()
    try:
        if _is_data_uri_image(s2):
            return _decode_data_uri_image(s2), "base64"
        if _is_http_url(s2):
            return _decode_url_image(s2), "url"
        if _is_file_uri(s2):
            return _decode_file_uri_image(s2), "file"
    except Exception:
        # If image loading fails, fall back to text per design.md
        return None, "text"
    return None, "text"


def _estimate_prompt_tokens(text: str) -> int:
    # Keep simple/deterministic; OpenAI-compatible field but not exact.
    # Tokenizers vary by model; a whitespace estimate is sufficient here.
    return max(1, len(text.strip().split())) if text.strip() else 0


def _embed_siglip2(engine: BaseInferenceEngine, model_name: str, item: str) -> Tuple[List[float], int]:
    def _get_output_tensor(outputs: Any, key: str) -> torch.Tensor:
        if hasattr(outputs, key):
            return getattr(outputs, key)
        if isinstance(outputs, dict) and key in outputs:
            return outputs[key]
        raise RuntimeError(f"SigLIP2 output missing {key}")

    img, _kind = _try_load_image_from_input(item)

    if img is not None:
        # Prefer image-only processing; some ONNX exports may require both
        # modalities, so fall back if needed.
        try:
            processed = engine.call_processor(
                model_name,
                images=[img],
                return_tensors="pt",
            )
            outputs = engine.call_model(model_name, **processed)
        except Exception:
            processed = engine.call_processor(
                model_name,
                text=[""],
                images=[img],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            outputs = engine.call_model(model_name, **processed)

        vec = _get_output_tensor(outputs, "image_embeds")
        vec = F.normalize(vec, p=2, dim=-1)
        return vec[0].tolist(), 0

    # Prefer text-only processing; some ONNX exports may require both modalities.
    try:
        processed = engine.call_processor(
            model_name,
            text=[item],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = engine.call_model(model_name, **processed)
    except Exception:
        dummy_img = Image.new("RGB", (512, 512), (0, 0, 0))
        processed = engine.call_processor(
            model_name,
            text=[item],
            images=[dummy_img],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = engine.call_model(model_name, **processed)

    vec = _get_output_tensor(outputs, "text_embeds")
    vec = F.normalize(vec, p=2, dim=-1)
    return vec[0].tolist(), _estimate_prompt_tokens(item)


def _embed_dinov2(engine: BaseInferenceEngine, model_name: str, item: str) -> Tuple[List[float], int]:
    img, _kind = _try_load_image_from_input(item)
    if img is None:
        raise ValueError("DinoV2 is vision-only; expected image input")
    processed = engine.call_processor(
        model_name, images=img, return_tensors="pt")
    outputs = engine.call_model(model_name, **processed)
    if not hasattr(outputs, "last_hidden_state"):
        raise RuntimeError("DinoV2 model output missing last_hidden_state")
    cls = outputs.last_hidden_state[:, 0, :]
    cls = F.normalize(cls, p=2, dim=-1)
    return cls[0].tolist(), 0


# engine = ONNXInferenceEngine()
# todo: openvino has bug!
engine = TorchInferenceEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    engine.load_models()
    yield
    # Shutdown: Clean up resources if needed
    pass

app = FastAPI(title="Embedding Server", lifespan=lifespan)


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    return ModelListResponse(
        data=[
            ModelCard(id=model_id, created=engine.created_ts)
            for model_id in engine.loaded_model_names()
        ]
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(models_loaded=engine.loaded_model_names())


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Creates an embedding vector representing the input text or image.
    Compatible with OpenAI API format.
    """
    if request.encoding_format not in (None, "float"):
        raise HTTPException(
            status_code=400, detail="Only encoding_format='float' is supported")

    engine.ensure_model_available(request.model)

    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]

    data: List[EmbeddingObject] = []
    prompt_tokens = 0

    for i, item in enumerate(inputs):
        if not isinstance(item, str):
            raise HTTPException(
                status_code=400, detail="Each input item must be a string")

        try:
            if request.model == "siglip2":
                emb, toks = _embed_siglip2(engine, request.model, item)
            elif request.model == "dinov2":
                emb, toks = _embed_dinov2(engine, request.model, item)
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported model {request.model}")
            prompt_tokens += toks
            data.append(EmbeddingObject(embedding=emb, index=i))
        except HTTPException:
            raise
        except ValueError as e:
            # Malformed input for model or invalid local file usage
            raise HTTPException(
                status_code=400, detail=f"Input[{i}] error: {str(e)}")
        except PermissionError as e:
            raise HTTPException(
                status_code=400, detail=f"Input[{i}] error: {str(e)}")
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=400, detail=f"Input[{i}] error: {str(e)}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Inference error: {str(e)}")

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting Embedding Server...")
    uvicorn.run(app, host="0.0.0.0", port=3752)
    # emb(edding) s(erver)-> 3752 based on qwer keyboard
