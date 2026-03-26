---
title: AVadCLIP — Audio-Visual Anomaly Detection
emoji: 📡
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.19.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Real-time CCTV anomaly detection using CLIP + audio fusion with AI alert notifications
---

# AVadCLIP — Audio-Visual Anomaly Detection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces)

> **  B.Tech ECE, VIth Semester**
> IIIT Nagpur · Pratik Ranjan · Anveeksha Jangid · Aman Kanaujiya · Utkarsh Nayan

---

## What it does

AVadCLIP detects anomalies in surveillance video by fusing **visual** and **audio** signals using OpenAI's pretrained CLIP model. Unlike traditional systems that rely on visuals alone, AVadCLIP analyses both what the camera *sees* and what the microphone *hears* — catching events like explosions (massive audio spike), theft (suspiciously silent), or fights (simultaneous visual + audio burst).

When an anomaly is detected, an **AI Alert Agent** automatically classifies the event type, determines severity, and notifies the security owner via email — even if no one is watching the monitor.

---

## Features

- 🎬 **Upload any video** — extracts frames + audio automatically
- 🧠 **CLIP ViT-B/16** — frozen pretrained visual encoder from OpenAI via HuggingFace
- 🔊 **Custom Audio CNN** — mel-spectrogram → 512-D CLIP-space embedding
- ⚡ **Adaptive Fusion Module** — learned per-dimension audio gating (core innovation)
- 🎯 **MIL Anomaly Scorer** — trained with weak video-level labels only
- 🔍 **Zero-Shot Mode** — score via cosine similarity with natural language prompts
- 🚨 **AI Alert Agent** — classifies anomaly type, determines severity, fires email alert
- 📊 **Rich visualisations** — score timeline, energy chart, fusion weights, heatmap
- 🎭 **Interactive Demo** — simulate fire / fight / theft / explosion without a video

---

## Model Architecture

```
VIDEO ──→ CLIP ViT-B/16 (FROZEN) ──→ Temporal 1D Conv ──→ visual_feat [T, 512]
                                                                  │
AUDIO ──→ Mel Spectrogram ──→ 4-block CNN ──→ MLP Proj ──→ audio_feat [T, 512]
                                                                  │
                        Adaptive Fusion:  fused = visual + sigmoid(W) ⊙ R
                                                                  │
                        MIL Scorer MLP (512→256→128→1) ──→ score ∈ [0,1] / frame
```

| Component | Details |
|-----------|---------|
| Visual encoder | `openai/clip-vit-base-patch16` — frozen |
| Audio encoder | Custom CNN (32→64→128→256 ch) + MLP projection |
| Temporal context | 1-D depthwise conv over frame sequence |
| Fusion | Sigmoid-gated cross-modal residual (per-dimension weights) |
| Scorer | MLP with MIL ranking loss — no frame-level labels needed |

---

## Alert System

Once an anomaly is detected the built-in alert agent:

1. Classifies the event type — **fire, fighting, theft, explosion, vandalism**
2. Assigns severity — **MEDIUM / HIGH / CRITICAL** based on confidence score
3. Sends an HTML email to the security owner with incident details and recommended action
4. Logs all incidents in the **🚨 Alert Dashboard** tab

### Enabling email alerts

Add these as **Space Secrets** (Settings → Variables and secrets):

| Secret | Description |
|--------|-------------|
| `SMTP_USER` | Your Gmail address |
| `SMTP_PASSWORD` | Gmail App Password (16-char) |
| `ALERT_TO` | Security owner's email |
| `COMPANY_NAME` | Your organisation name |
| `OWNER_NAME` | Owner's display name |

---

## Tech Stack

| Technology | Role |
|------------|------|
| Python 3.10 | Core language |
| PyTorch 2.1 | Deep learning framework |
| HuggingFace Transformers | CLIP pretrained model |
| OpenCV | Video frame extraction |
| Librosa | Audio extraction + mel spectrogram |
| Gradio 4 | Web interface + HuggingFace Spaces deployment |
| Matplotlib | Visualisations |
| smtplib (stdlib) | Email alert delivery |

---

## Local Setup

```bash
git clone https://https://huggingface.co/spaces/Utkarsh430/Ai_agent_audio_video_anomally_detection
cd AVadCLIP
pip install -r requirements.txt
python app.py
# Open https://huggingface.co/spaces/Utkarsh430/Ai_agent_audio_video_anomally_detection
```

---

## References

- Wu et al. (2025) — *AVadCLIP: Audio-Visual Adaptive CLIP for Weakly-Supervised Video Anomaly Detection*
- Radford et al. (2021) — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Sultani et al. (2018) — *Real-world Anomaly Detection in Surveillance Videos* (MIL framework)

---

## License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

© 2025 Pratik Ranjan, Anveeksha Jangid, Aman Kanaujiya, Utkarsh Nayan — IIIT Nagpur
