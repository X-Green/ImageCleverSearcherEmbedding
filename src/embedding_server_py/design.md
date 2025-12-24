# Embedding Server Design Document

## 1. Overview
The Embedding Server is a high-performance, local API service designed to convert text and images into vector representations (embeddings). It leverages **ONNX Runtime** for efficient inference across various hardware backends (CPU, NVIDIA GPU, Windows DirectML) and provides an **OpenAI-compatible API** for seamless integration with existing tools and frameworks.

## 2. Core Technologies
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous API)
- **Inference Engine**: [ONNX Runtime](https://onnxruntime.ai/) via [Hugging Face Optimum](https://huggingface.co/docs/optimum/index)
- **Image Processing**: [Pillow (PIL)](https://python-pillow.org/)
- **Environment Management**: [uv](https://github.com/astral-sh/uv) (Fast Python package installer and resolver)

## 3. Models
The server supports two primary model architectures, selectable via the API:

### 3.1 SigLIP 2 (Multimodal)
- **Model**: `google/siglip2-base-patch16-512` alias `siglip2`
- **Capabilities**: Text-to-Image, Image-to-Image, and Text-to-Text retrieval.
- **Implementation**: Uses `ORTModelForZeroShotImageClassification` to extract features from both vision and text towers.
- **Normalization**: Embeddings are L2-normalized to ensure cosine similarity compatibility.

### 3.2 DinoV2 (Vision-Only)
- **Model**: `facebook/dinov2-base` alias `dinov2`
- **Capabilities**: Pure vision tasks, image clustering, and image-to-image retrieval.
- **Implementation**: Uses `ORTModelForFeatureExtraction`.
- **Feature Extraction**: Extracts the `[CLS]` token from the last hidden state.
- **Normalization**: Embeddings are L2-normalized.

## 4. API Specification
The server implements the OpenAI v1 Embeddings interface.

### 4.1 Endpoint: `GET /v1/models`
**Response Body:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "siglip2",
      "object": "model",
      "created": 1700000000,
      "owned_by": "local",
      "permission": []
    },
    {
      "id": "dinov2",
      "object": "model",
      "created": 1700000000,
      "owned_by": "local",
      "permission": []
    }
  ]
}
```

### 4.2 Endpoint: `GET /health`
**Response Body:**
```json
{
  "status": "ok",
  "models_loaded": ["siglip2", "dinov2"]
}
```


### 4.3 Endpoint: `POST /v1/embeddings`
**Request Body:**
```json
{
  "input": ["A photo of a cat", "data:image/png;base64,..."],
  "model": "siglip2",
  "encoding_format": "float"
}
```

**Response Body:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0234, ...],
      "index": 0
    }
  ],
  "model": "siglip2",
  "usage": {
    "prompt_tokens": 2,
    "total_tokens": 2
  }
}
```

### 4.4 Input Recognition Logic
The server automatically detects the type of input in the `input` array:
1.  **Base64 Image**: Starts with `data:image/{format};base64,`.
2.  **URL**: Starts with `http://` or `https://`.
3.  **Local Path**: Starts with `file:///`. Note: Enabled **only if** local server configuration allows it, for security reasons.
4.  **Plain Text**: Fallback if none of the above match or if image loading fails.

Note: images and text can be mixed in the same request for **multimodal models** (SigLIP2). Separate embeddings will be returned in the same order. Vision-only models (DinoV2) will return a 400 error for text inputs.

## 5. Inference Engine Details
### 5.1 Execution Providers
The server prioritizes hardware acceleration in the following order:
1.  `CUDAExecutionProvider`: NVIDIA GPUs.
2.  `DmlExecutionProvider`: Windows DirectML (AMD/Intel/NVIDIA GPUs).
3.  `CPUExecutionProvider`: Fallback for all systems.

### 5.2 Optimization
- **Graph Optimization**: `GraphOptimizationLevel.ORT_ENABLE_ALL` is enabled for maximum performance.
- **Precision**: Models are exported in FP32 by default; FP16/INT8 quantization can be applied for edge devices.

## 6. Implementation Strategy
### 6.1 Preprocessing
- **Images**: Handled by `AutoProcessor`. Includes resizing (e.g., 512x512 for SigLIP 2), center cropping, and normalization (mean/std).
- **Text**: Handled by `AutoTokenizer`. Includes padding and truncation to the model's maximum sequence length.

### 6.2 Error Handling
- `400 Bad Request`: Unsupported model or malformed input.
- `404 Not Found`: Specified model path not found.
- `500 Internal Server Error`: Inference failure or model loading issues.

## 7. Deployment & Installation
### 7.1 Environment Setup
The project uses `uv` for near-instant environment creation and dependency installation.
- **Windows**: `install.ps1` automates `uv` installation, venv creation, and dependency setup.

### 7.2 Model Setup
A dedicated script `setup_models.py` is used to:
1. Download models from Hugging Face.
2. Convert them to ONNX format using `optimum-cli`.
3. Store them in the `models/` directory.
