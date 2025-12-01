# Multi-Agent AI Content Studio

An autonomous video content analysis pipeline that processes an arbitrary input video and produces three deliverables: timestamped captions, highlight segments, and high-quality thumbnails. The system is implemented as a coordinated multi-agent architecture using only open-source models and runs entirely on a free Google Colab T4 GPU instance.

## Architecture Overview

The pipeline consists of five specialized perception agents orchestrated by a central coordinator:

| Agent                  | Model / Implementation                                   | Function                                                                 |
|------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------|
| Transcription Agent   | OpenAI Whisper-small (244 M)                                    | Word-level timestamped transcription, SRT export                         |
| Semantic Agent         | Google SigLIP-so400m-patch14-384 (400 M vision encoder)         | Frame-level semantic embeddings; cosine distance for scene/topic change |
| Visual Interest Agent  | Custom lightweight module (Sobel edge density + center bias + blob contrast) | Proxy for object/activity detection; replaces YOLOv8n to avoid PyTorch 2.6+ unpickling issues |
| Audio Energy Agent     | Librosa RMS energy with sliding window smoothing               | Detects high-energy speech or music segments                             |
| Segmentation & Thumbnail Agent | rembg (U²-Net / IS-Net backbone, ONNX) + PIL post-processing | Foreground subject isolation, background removal, contrast/color enhancement |

All heavy perception tasks (embedding extraction, visual interest scoring, segmentation) are executed in memory-safe batches with explicit CUDA cache clearing to stay within the 16 GB T4 limit.

## Multi-Modal Highlight Scoring

Highlight confidence at frame t is computed as:

score(t) = Δsemantic(t) × (0.6 + 0.4 × visual_interest(t)) × (0.7 + 0.3 × audio_energy(t))


where Δsemantic is the normalized SigLIP distance to the previous keyframe. The score is smoothed with a uniform filter (size = 9). Peaks above the 75th percentile (minimum inter-peak distance 20 keyframes) are selected as highlight centers; 10-second clips are extracted around each.

## Key Implementation Details

- Keyframe extraction: 1 FPS with frame-difference thresholding to reduce processing load (~100–350 frames for typical videos)
- SigLIP embeddings: vision tower only, batch size 8, L2-normalized CLS token
- Background removal: rembg with GPU-accelerated ONNX runtime
- Thumbnail variants: transparent PNG, blurred-background composite, high-contrast 1280×720 version with optional border
- All intermediate results (embeddings, timestamps, word-level JSON, highlight metadata) are persisted for inspection and future extension

## Limitations & Upgrade Path

- Visual Interest Agent is a lightweight substitute for YOLOv8n due to PyTorch 2.6+ weights_only restrictions in Colab
- Segmentation uses rembg (U²-Net/IS-Net) instead of BiRefNet for identical runtime constraints

Both components can be restored to the original YOLOv8n and BiRefNet models by running on an environment with PyTorch ≤2.4 or by adding appropriate safe_globals.

## Requirements

Runs on Google Colab Free Tier (T4 GPU). Dependencies:
```text
torch, torchvision, whisper, moviepy, ultralytics, transformers, rembg[gpu], onnxruntime-gpu, librosa, opencv-python, pillow, matplotlib, scipy
