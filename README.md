# AVadCLIP — Audio-Visual Anomaly Detection System

<div align="center">

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Utkarsh430/Ai_agent_audio_video_anomally_detection)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)

**Real-time CCTV anomaly detection using CLIP + audio fusion with AI-powered alert notifications**

</div>

---

## The Problem We Solve

Traditional surveillance systems depend on human operators watching screens 24/7 — a task that is cognitively exhausting, expensive, and error-prone. Even modern AI-based systems analyse only video frames, missing critical **audio signals** that uniquely identify events like explosions (massive audio spike), theft (suspicious silence), or riots (simultaneous audio + visual burst).

**AVadCLIP bridges this gap** — the first open-source implementation of the Wu et al. (2025) paper that fuses CLIP visual embeddings with a custom audio CNN in a learnable adaptive fusion module, achieving richer anomaly representations than either modality alone.

---

## 🎯 Key Highlights

| | What We Built |
|---|---|
| 🧠 | **Novel fusion architecture** — per-dimension sigmoid gating over CLIP visual + audio embeddings |
| 🎬 | **Weakly supervised** — trained with video-level labels only; no expensive frame-level annotations |
| 🔍 | **Zero-shot mode** — classify unseen anomaly types with plain English prompts, no retraining |
| 🚨 | **End-to-end alert pipeline** — AI classifies event, determines severity, sends HTML email instantly |
| ⚡ | **Deployable in minutes** — single `app.py`, runs free on HuggingFace Spaces CPU tier |
| 📊 | **Rich visualisations** — score timeline, energy chart, fusion weights, anomaly heatmap |

---

## 🏗️ Model Architecture

```
VIDEO ──→ CLIP ViT-B/16 (FROZEN, HuggingFace) ──→ Temporal 1D Conv ──→ visual_feat [T, 512]
                                                                                │
AUDIO ──→ Mel Spectrogram ──→ 4-block CNN ──→ MLP Projection ──→ audio_feat [T, 512]
                                                                                │
                     ┌──────────────── Adaptive Fusion ─────────────────┐
                     │   joint = cat([visual, audio])  → [B, T, 1024]   │
                     │   W     = sigmoid(Linear(joint)) → [B, T, 512]   │  ← per-dimension audio gate
                     │   R     = MLP(joint)             → [B, T, 512]   │  ← cross-modal residual
                     │   fused = LayerNorm(visual + W ⊙ R)              │
                     └──────────────────────────────────────────────────┘
                                                │
                     MIL Scorer MLP (512→256→128→1) ──→ score ∈ [0,1] per frame
```

### Component Breakdown

| Component | Architecture | Innovation |
|-----------|-------------|------------|
| **Visual Encoder** | `openai/clip-vit-base-patch16` — 12-layer ViT, frozen | Leverages 400M-pair CLIP pretraining |
| **Audio Encoder** | Custom CNN (32→64→128→256 ch) + BatchNorm + MLP projection | Maps mel-spectrograms to CLIP embedding space |
| **Temporal Context** | 1-D depthwise conv over frame sequence | Adds motion awareness without extra parameters |
| **Adaptive Fusion** | Sigmoid-gated cross-modal residual — **per-dimension** | Core paper contribution; richer than scalar gating |
| **MIL Scorer** | 3-layer MLP with dropout | Trained with ranking loss; no frame annotations needed |

---

## 🚨 AI Alert Agent

When an anomaly is detected, the built-in alert agent autonomously:

1. **Classifies** the event type — fire, fighting, theft, explosion, vandalism — using visual/audio energy ratios
2. **Assigns severity** — MEDIUM / HIGH / CRITICAL based on confidence score
3. **Sends an HTML email** to the configured security owner with incident ID, camera location, confidence score, and a recommended response action
4. **Logs all incidents** in a live Alert Dashboard tab with stats (total / critical / high / medium counts)

This transforms AVadCLIP from a model into a **complete security automation product**.

---

## 🔍 Scoring Modes

**MIL Mode (trained head)** — Uses the trained MLP scorer. High precision once a checkpoint is fine-tuned on domain footage.

**Zero-Shot Mode (CLIP text)** — Computes cosine similarity between fused audio-visual embeddings and natural language prompts like `"a video of someone stealing"`. Works immediately with **zero training**, enabling detection of novel anomaly types on the fly.

Combined score = (MIL score + zero-shot score) / 2, giving the best of both worlds.

---

## 📊 Visualisations

Every analysis run produces four interactive charts:

- **Score Timeline** — frame-level anomaly scores with threshold line, flagged frame markers, and peak annotation
- **Visual vs Audio Energy** — L2 norms of CLIP and audio embeddings overlaid; spikes reveal cross-modal anomaly signatures
- **Fusion Weights** — bar chart of the learned audio gate W per frame; shows exactly how much the model trusted audio at each moment
- **Anomaly Heatmap** — colour-coded strip from deep blue (safe) to red (anomalous) for instant situational overview

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.1, HuggingFace Transformers |
| Visual Backbone | CLIP ViT-B/16 (`openai/clip-vit-base-patch16`) |
| Audio Processing | Librosa (mel-spectrogram), torchaudio |
| Video I/O | OpenCV (frame extraction) |
| Web Interface | Gradio 4 (HuggingFace Spaces compatible) |
| Visualisation | Matplotlib (dark surveillance-monitor theme) |
| Alert Delivery | Python stdlib `smtplib` — zero extra dependencies |
| Deployment | HuggingFace Spaces (free CPU tier) |

---

## 🚀 Quick Start

### Try the live demo — no setup needed
**👉 [https://huggingface.co/spaces/Utkarsh430/Ai_agent_audio_video_anomally_detection](https://huggingface.co/spaces/Utkarsh430/Ai_agent_audio_video_anomally_detection)**

### Run locally

```bash
git clone https://huggingface.co/spaces/Utkarsh430/Ai_agent_audio_video_anomally_detection
cd AVadCLIP
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

### Deploy your own HuggingFace Space

```
1. Go to https://huggingface.co/spaces → New Space → SDK: Gradio
2. Upload app.py and requirements.txt
3. (Optional) Add Space Secrets for email alerts
4. Your public URL is live in ~2 minutes
```

### Enable email alerts

Add these as **Space Secrets** (Settings → Variables and secrets):

| Secret | Description |
|--------|-------------|
| `SMTP_USER` | Your Gmail address |
| `SMTP_PASSWORD` | Gmail App Password (16-char) |
| `ALERT_TO` | Security owner's email |
| `COMPANY_NAME` | Your organisation name |
| `OWNER_NAME` | Owner's display name |

---

## 📦 Requirements

```
gradio>=4.19.0
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
transformers>=4.37.0
librosa>=0.10.1
opencv-python-headless>=4.8.0
matplotlib>=3.8.0
numpy>=1.24.0
```

No LangChain, no external alert APIs — the full agent runs on Python stdlib.

---

## 📄 References

- **Wu et al. (2025)** — *AVadCLIP: Audio-Visual Adaptive CLIP for Weakly-Supervised Video Anomaly Detection*
- **Radford et al. (2021)** — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP) — [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- **Sultani et al. (2018)** — *Real-world Anomaly Detection in Surveillance Videos* (MIL framework) — [arXiv:1801.04264](https://arxiv.org/abs/1801.04264)

---

## 📜 License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.
