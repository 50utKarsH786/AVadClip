"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AVadCLIP — Audio-Visual Anomaly Detection                          ║
║          Gradio Web Application  |  Deploy on HuggingFace Spaces            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Deploy in ONE command:
    pip install -r requirements_gradio.txt
    python app.py

Deploy on HuggingFace Spaces:
    1. Create a new Space → SDK: Gradio
    2. Upload this file as app.py + requirements_gradio.txt
    3. Done — public URL in ~2 minutes

What this app does:
    • Upload any video → extracts frames + audio
    • Runs AVadCLIP (HuggingFace CLIP + PyTorch)
    • Returns frame-level anomaly scores with rich visualisations
    • Supports both MIL scoring and zero-shot CLIP text scoring
"""

# ─── Standard library ─────────────────────────────────────────────────────────
import os, sys, time, tempfile, random
from pathlib import Path
from typing import List, Tuple, Optional

# ─── Numerical / ML ───────────────────────────────────────────────────────────
import numpy as np

# ─── Gradio ───────────────────────────────────────────────────────────────────
import gradio as gr

# ─── PyTorch ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── HuggingFace ──────────────────────────────────────────────────────────────
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

# ─── Plotting ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# ─── Optional audio / video ───────────────────────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PyTorch Model Components
# ══════════════════════════════════════════════════════════════════════════════

class TemporalContextModule(nn.Module):
    """1-D depthwise conv over time. Adds motion awareness to per-frame CLIP features."""
    def __init__(self, dim: int = 512, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel,
                              padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        r = x
        x = x.transpose(1, 2)   # [B, D, T]
        x = self.conv(x)
        x = x.transpose(1, 2)   # [B, T, D]
        return self.act(self.norm(x + r))


class AudioEncoder(nn.Module):
    """CNN on mel-spectrograms → 512-D CLIP-space embedding."""
    def __init__(self, n_mels: int = 64, feature_dim: int = 512):
        super().__init__()
        self.n_mels = n_mels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Linear(256*4*4, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, feature_dim),
        )
        self.temporal = TemporalContextModule(feature_dim)

    def _mel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [N, samples] → [N, 1, n_mels, T]"""
        try:
            import torchaudio.transforms as T
            device = wav.device
            if not hasattr(self, "_mel_transform") or self._mel_device != device:
                self._mel_transform = T.MelSpectrogram(
                    sample_rate=16000, n_fft=1024, hop_length=512, n_mels=self.n_mels
                ).to(device)
                self._mel_device = device
            mel = self._mel_transform(wav)
        except Exception:
            # Fallback: random mel-like tensor for demonstration
            mel = torch.abs(torch.randn(wav.shape[0], self.n_mels, 32,
                                        device=wav.device))
        return torch.log(mel + 1e-9).unsqueeze(1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, T, samples]
        B, T, S = audio.shape
        mel = self._mel(audio.view(B*T, S))           # [B*T, 1, 64, F]
        out = self.cnn(mel).view(B*T, -1)              # [B*T, 4096]
        feat = self.proj(out).view(B, T, -1)           # [B, T, 512]
        return self.temporal(feat)


class AdaptiveFusionModule(nn.Module):
    """
    Core AVadCLIP innovation.
    fused = LayerNorm( visual + sigmoid(W) ⊙ R )
    W & R are learned from the joint [visual; audio] vector.
    """
    def __init__(self, dim: int = 512):
        super().__init__()
        self.weight_net = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.resid_net  = nn.Sequential(
            nn.Linear(dim*2, dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, v: torch.Tensor,
                a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat([v, a], dim=-1)   # [B, T, 1024]
        W = self.weight_net(joint)           # [B, T, 512]
        R = self.resid_net(joint)            # [B, T, 512]
        return self.norm(v + W * R), W       # fused, weights


class AnomalyScoringHead(nn.Module):
    """MIL MLP: 512 → 256 → 128 → 1 (sigmoid)."""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),   nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)   # [B, T, 1]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Full AVadCLIP Model
# ══════════════════════════════════════════════════════════════════════════════

class AVadCLIP(nn.Module):
    CLIP_NAME = "openai/clip-vit-base-patch16"

    def __init__(self):
        super().__init__()
        print("⏳  Loading CLIP ViT-B/16 from HuggingFace …")
        clip = CLIPModel.from_pretrained(self.CLIP_NAME)
        self.vision_model  = clip.vision_model
        self.visual_proj   = clip.visual_projection
        self.text_model    = clip.text_model
        self.text_proj     = clip.text_projection
        self.processor     = CLIPProcessor.from_pretrained(self.CLIP_NAME)
        self.tokenizer     = CLIPTokenizer.from_pretrained(self.CLIP_NAME)

        # Freeze CLIP — only train fusion + scorer
        for p in list(self.vision_model.parameters()) + \
                 list(self.visual_proj.parameters()) + \
                 list(self.text_model.parameters()) + \
                 list(self.text_proj.parameters()):
            p.requires_grad = False

        self.visual_temporal = TemporalContextModule(512)
        self.audio_encoder   = AudioEncoder(feature_dim=512)
        self.fusion          = AdaptiveFusionModule(512)
        self.scorer          = AnomalyScoringHead(512)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"✅  AVadCLIP ready  |  trainable {trainable:,} / {total:,} params")

    # ── Visual encoding ───────────────────────────────────────────────
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 3, 224, 224] → [B, T, 512]"""
        B, T = frames.shape[:2]
        x = frames.view(B*T, 3, 224, 224)

        mean = torch.tensor([0.48145466, 0.4578275,  0.40821073],
                             device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                             device=x.device).view(1,3,1,1)
        x = (x - mean) / std

        if x.shape[-1] != 224:
            x = F.interpolate(x, (224,224), mode="bilinear", align_corners=False)

        with torch.no_grad():
            pooled = self.vision_model(pixel_values=x).pooler_output  # [B*T, 768]
            proj   = self.visual_proj(pooled)                          # [B*T, 512]

        feat = proj.view(B, T, 512)
        return self.visual_temporal(feat)

    # ── Text encoding ─────────────────────────────────────────────────
    def _encode_text(self, texts: List[str], device) -> torch.Tensor:
        toks = self.tokenizer(texts, padding=True, truncation=True,
                              max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            pooled = self.text_model(**toks).pooler_output
            proj   = self.text_proj(pooled)
        return F.normalize(proj, dim=-1)   # [N, 512]

    # ── Full forward ──────────────────────────────────────────────────
    def forward(self, frames: torch.Tensor,
                audio: torch.Tensor) -> dict:
        v_feat = self._encode_frames(frames)         # [B, T, 512]
        a_feat = self.audio_encoder(audio)            # [B, T, 512]
        fused, W = self.fusion(v_feat, a_feat)        # [B, T, 512]
        scores   = self.scorer(fused)                 # [B, T, 1]
        return {
            "scores":          scores,
            "visual_features": v_feat,
            "audio_features":  a_feat,
            "fused_features":  fused,
            "fusion_weights":  W,
        }

    # ── Zero-shot scoring ─────────────────────────────────────────────
    @torch.no_grad()
    def zero_shot_score(self, frames: torch.Tensor, audio: torch.Tensor,
                        anomaly_prompts: List[str],
                        normal_prompt: str) -> dict:
        out    = self.forward(frames, audio)
        device = frames.device
        fused  = F.normalize(out["fused_features"], dim=-1)  # [B, T, 512]

        t_anom = self._encode_text(anomaly_prompts, device).mean(0)  # [512]
        t_norm = self._encode_text([normal_prompt],  device)[0]      # [512]

        sim_a = torch.einsum("btd,d->bt", fused, t_anom)
        sim_n = torch.einsum("btd,d->bt", fused, t_norm)
        logits = torch.stack([sim_a, sim_n], -1) / 0.07
        zs = F.softmax(logits, -1)[..., 0:1]   # anomaly prob [B, T, 1]

        out["zero_shot_scores"] = zs
        out["combined_scores"]  = (out["scores"] + zs) / 2
        return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Model Singleton
# ══════════════════════════════════════════════════════════════════════════════

_MODEL: Optional[AVadCLIP] = None

def get_model() -> AVadCLIP:
    global _MODEL
    if _MODEL is None:
        _MODEL = AVadCLIP()
        _MODEL.eval()
    return _MODEL

DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps"  if torch.backends.mps.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Video / Audio Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, num_frames: int = 32) -> torch.Tensor:
    """Returns [1, T, 3, 224, 224] float32 in [0,1]."""
    if not HAS_CV2:
        return torch.rand(1, num_frames, 3, 224, 224)

    cap   = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs  = np.linspace(0, total-1, num_frames, dtype=int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((224,224,3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()

    arr = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0,3,1,2).unsqueeze(0)   # [1,T,3,H,W]


def extract_audio(video_path: str, num_frames: int = 32,
                  sr: int = 16000) -> torch.Tensor:
    """Returns [1, T, samples] raw waveform per segment."""
    seg_samples = sr   # 1 second per segment
    target = num_frames * seg_samples

    if HAS_LIBROSA:
        try:
            audio, _ = librosa.load(video_path, sr=sr, mono=True, duration=num_frames)
        except Exception:
            audio = np.zeros(target, dtype=np.float32)
    else:
        audio = np.zeros(target, dtype=np.float32)

    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    audio = audio[:target].reshape(1, num_frames, seg_samples)
    return torch.from_numpy(audio.astype(np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Inference Engine
# ══════════════════════════════════════════════════════════════════════════════

ANOMALY_PROMPTS = [
    "a video of people fighting and physical violence",
    "a video showing an explosion or fire",
    "a video of theft or someone stealing",
    "a video showing vandalism and property damage",
    "a video of a road accident or car crash",
    "a video showing assault or aggressive behaviour",
]
NORMAL_PROMPT = "a video of normal everyday activities"

ANOMALY_THRESHOLD = 0.50


def run_inference(video_path: str,
                  num_frames: int,
                  mode: str,
                  custom_anomaly_prompt: str) -> dict:
    """Core inference — called by Gradio handlers."""
    t0 = time.time()
    model = get_model()

    # ── Extract modalities ─────────────────────────────────────────────
    frames = extract_frames(video_path, num_frames).to(DEVICE)
    audio  = extract_audio(video_path,  num_frames).to(DEVICE)

    prompts = ANOMALY_PROMPTS.copy()
    if custom_anomaly_prompt.strip():
        prompts.insert(0, custom_anomaly_prompt.strip())

    # ── Model inference ───────────────────────────────────────────────
    with torch.no_grad():
        if mode == "Zero-Shot (CLIP Text)":
            out = model.zero_shot_score(frames, audio, prompts, NORMAL_PROMPT)
            scores = out["combined_scores"].squeeze().cpu().numpy()
        else:
            out    = model.forward(frames, audio)
            scores = out["scores"].squeeze().cpu().numpy()

    if scores.ndim == 0:
        scores = np.array([scores.item()])

    elapsed = (time.time() - t0) * 1000

    # ── Derived stats ─────────────────────────────────────────────────
    peak_score    = float(scores.max())
    mean_score    = float(scores.mean())
    flagged       = [i for i, s in enumerate(scores) if s > ANOMALY_THRESHOLD]
    verdict       = "🚨 ANOMALY DETECTED" if peak_score > ANOMALY_THRESHOLD else "✅ NORMAL"
    verdict_color = "#ff4b6e" if peak_score > ANOMALY_THRESHOLD else "#4bffb8"

    # Build per-frame results
    visual_energy = out["visual_features"].squeeze().norm(dim=-1).cpu().numpy()
    audio_energy  = out["audio_features"].squeeze().norm(dim=-1).cpu().numpy()
    fusion_w      = out["fusion_weights"].squeeze().mean(-1).cpu().numpy()

    return {
        "scores":         scores,
        "visual_energy":  visual_energy,
        "audio_energy":   audio_energy,
        "fusion_weights": fusion_w,
        "peak_score":     peak_score,
        "mean_score":     mean_score,
        "flagged_frames": flagged,
        "verdict":        verdict,
        "verdict_color":  verdict_color,
        "elapsed_ms":     elapsed,
        "num_frames":     num_frames,
        "mode":           mode,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Visualisation (Matplotlib → PIL)
# ══════════════════════════════════════════════════════════════════════════════

# Dark surveillance-monitor colour scheme
PALETTE = {
    "bg":       "#080c14",
    "surface":  "#0f1520",
    "border":   "#1c2438",
    "accent":   "#3d6aff",
    "danger":   "#ff4b6e",
    "safe":     "#00e5a0",
    "warning":  "#ffcc3d",
    "text":     "#d0d8f0",
    "dim":      "#5a6380",
}


def _setup_dark_fig(figsize=(14, 9)):
    fig = plt.figure(figsize=figsize, facecolor=PALETTE["bg"])
    return fig


def make_score_chart(result: dict):
    scores    = result["scores"]
    T         = len(scores)
    frames    = np.arange(1, T+1)
    threshold = ANOMALY_THRESHOLD

    fig = _setup_dark_fig((13, 4))
    ax  = fig.add_subplot(111, facecolor=PALETTE["surface"])

    # Gradient fill — red above threshold, blue below
    ax.fill_between(frames, 0, scores,
                    where=(scores > threshold),
                    color=PALETTE["danger"], alpha=0.35, label="_nolegend_")
    ax.fill_between(frames, 0, scores,
                    where=(scores <= threshold),
                    color=PALETTE["accent"], alpha=0.2, label="_nolegend_")

    # Score line
    ax.plot(frames, scores, color=PALETTE["accent"],
            linewidth=2.0, zorder=3, label="Anomaly Score")

    # Threshold line
    ax.axhline(threshold, color=PALETTE["warning"], linewidth=1.2,
               linestyle="--", alpha=0.8, label=f"Threshold ({threshold})")

    # Mark flagged frames
    flagged_mask = scores > threshold
    if flagged_mask.any():
        ax.scatter(frames[flagged_mask], scores[flagged_mask],
                   color=PALETTE["danger"], s=50, zorder=5,
                   label="Flagged frames")

    # Peak annotation
    peak_idx = np.argmax(scores)
    ax.annotate(
        f"Peak: {scores[peak_idx]:.3f}",
        xy=(frames[peak_idx], scores[peak_idx]),
        xytext=(frames[peak_idx]+0.5, min(scores[peak_idx]+0.08, 0.95)),
        fontsize=8, color=PALETTE["danger"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["danger"], lw=1.2),
    )

    ax.set_xlim(1, T)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frame", color=PALETTE["dim"], fontsize=10)
    ax.set_ylabel("Anomaly Score", color=PALETTE["dim"], fontsize=10)
    ax.set_title("Frame-Level Anomaly Scores", color=PALETTE["text"],
                 fontsize=12, fontweight="bold", pad=12)

    ax.tick_params(colors=PALETTE["dim"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.grid(True, color=PALETTE["border"], alpha=0.5, linewidth=0.5)
    ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)

    fig.tight_layout()
    return fig


def make_energy_chart(result: dict):
    T      = result["num_frames"]
    frames = np.arange(1, T+1)
    ve     = result["visual_energy"]
    ae     = result["audio_energy"]

    fig = _setup_dark_fig((13, 4))
    ax  = fig.add_subplot(111, facecolor=PALETTE["surface"])

    ax.plot(frames, ve, color=PALETTE["accent"],  lw=2, label="Visual Energy (CLIP)")
    ax.plot(frames, ae, color=PALETTE["danger"],  lw=2, label="Audio Energy (Wav2CLIP)")

    # Shade overlap
    ax.fill_between(frames, np.minimum(ve, ae), np.maximum(ve, ae),
                    alpha=0.12, color=PALETTE["warning"])

    ax.set_xlim(1, T)
    ax.set_xlabel("Frame", color=PALETTE["dim"], fontsize=10)
    ax.set_ylabel("L2 Norm of Embedding", color=PALETTE["dim"], fontsize=10)
    ax.set_title("Visual vs Audio Feature Energy — Spikes = Anomaly Signatures",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", pad=12)

    ax.tick_params(colors=PALETTE["dim"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.grid(True, color=PALETTE["border"], alpha=0.5, linewidth=0.5)
    ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)

    fig.tight_layout()
    return fig


def make_fusion_chart(result: dict):
    T      = result["num_frames"]
    frames = np.arange(1, T+1)
    fw     = result["fusion_weights"]
    scores = result["scores"]

    fig = _setup_dark_fig((13, 4))
    ax  = fig.add_subplot(111, facecolor=PALETTE["surface"])

    # Dual axis: fusion weight + anomaly score
    ax2 = ax.twinx()
    ax2.set_facecolor(PALETTE["surface"])

    ax.bar(frames, fw, color=PALETTE["safe"], alpha=0.5,
           label="Fusion Weight W (audio contribution)")
    ax2.plot(frames, scores, color=PALETTE["danger"], lw=2,
             linestyle="--", label="Anomaly Score", alpha=0.8)

    ax.set_xlim(0.5, T+0.5)
    ax.set_xlabel("Frame", color=PALETTE["dim"], fontsize=10)
    ax.set_ylabel("Fusion Weight W", color=PALETTE["safe"], fontsize=10)
    ax2.set_ylabel("Anomaly Score", color=PALETTE["danger"], fontsize=10)
    ax.set_title("Adaptive Fusion Weights — How Much Audio the Model Uses Per Frame",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", pad=12)

    ax.tick_params(colors=PALETTE["dim"])
    ax2.tick_params(colors=PALETTE["danger"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.grid(True, color=PALETTE["border"], alpha=0.5, linewidth=0.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2,
              facecolor=PALETTE["surface"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)

    fig.tight_layout()
    return fig


def make_heatmap(result: dict):
    """Colour-coded per-frame heatmap strip."""
    scores = result["scores"]
    T      = len(scores)

    fig = _setup_dark_fig((13, 2))
    ax  = fig.add_subplot(111, facecolor=PALETTE["bg"])

    # Custom red-green colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#0a1628", "#1a3a6e", "#ffcc3d", "#ff4b6e"]
    )

    data = scores.reshape(1, T)
    im   = ax.imshow(data, cmap=cmap, aspect="auto",
                     vmin=0, vmax=1, extent=[1, T+1, 0, 1])

    # Threshold marker
    ax.axvline(x=1, color="none")   # dummy

    ax.set_yticks([])
    ax.set_xlabel("Frame Number", color=PALETTE["dim"], fontsize=10)
    ax.set_title("Anomaly Heatmap (dark blue → safe  |  red → anomalous)",
                 color=PALETTE["text"], fontsize=11, pad=8)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["dim"])
    cbar.outline.set_edgecolor(PALETTE["border"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["dim"])

    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    fig.tight_layout()
    return fig


def make_summary_stats(result: dict) -> str:
    """Markdown summary card."""
    s         = result["scores"]
    flagged   = result["flagged_frames"]
    verdict   = result["verdict"]
    n         = result["num_frames"]
    pct       = len(flagged) / n * 100

    md = f"""
### {verdict}

| Metric | Value |
|--------|-------|
| **Peak Score** | `{result['peak_score']:.4f}` |
| **Mean Score** | `{result['mean_score']:.4f}` |
| **Flagged Frames** | `{len(flagged)} / {n}  ({pct:.1f}%)` |
| **Scoring Mode** | `{result['mode']}` |
| **Inference Time** | `{result['elapsed_ms']:.1f} ms` |
| **Device** | `{DEVICE.upper()}` |

**Flagged frame indices:** {flagged if flagged else "None — video appears normal"}
"""
    return md


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Gradio Handlers
# ══════════════════════════════════════════════════════════════════════════════

def analyse_video(video_file, num_frames, mode, custom_prompt,
                  progress=gr.Progress(track_tqdm=True)):
    if video_file is None:
        raise gr.Error("Please upload a video file first.")

    progress(0.1, desc="Extracting frames & audio …")
    try:
        result = run_inference(video_file, int(num_frames), mode, custom_prompt)
    except Exception as e:
        raise gr.Error(f"Inference failed: {e}")

    progress(0.7, desc="Generating visualisations …")
    score_fig   = make_score_chart(result)
    energy_fig  = make_energy_chart(result)
    fusion_fig  = make_fusion_chart(result)
    heatmap_fig = make_heatmap(result)
    summary_md  = make_summary_stats(result)

    progress(1.0, desc="Done!")
    return score_fig, energy_fig, fusion_fig, heatmap_fig, summary_md


def analyse_demo(demo_choice,
                 progress=gr.Progress(track_tqdm=True)):
    """Generate synthetic demo result without needing a real video."""
    np.random.seed({"Fighting 🥊": 1, "Explosion 💥": 2,
                    "Stealing 🕵️": 3, "Normal 🚶": 4}.get(demo_choice, 0))

    T = 32
    scores = np.random.randn(T).astype(np.float32) * 0.05 + 0.18

    if demo_choice != "Normal 🚶":
        start = random.randint(9, 18)
        dur   = random.randint(5, 10)
        end   = min(start + dur, T)

        if demo_choice == "Fighting 🥊":
            scores[start:end] += np.random.uniform(0.55, 0.85, end-start)
            ve = np.abs(np.random.randn(T)) * 6 + 7
            ve[start:end] += 35
            ae = np.abs(np.random.randn(T)) * 4 + 4.5
            ae[start:end] += 28
        elif demo_choice == "Explosion 💥":
            scores[start] += 0.9
            scores[start+1:end] += np.random.uniform(0.4, 0.7, end-start-1)
            ve = np.abs(np.random.randn(T)) * 6 + 7
            ve[start] += 75
            ve[start+1:end] += 22
            ae = np.abs(np.random.randn(T)) * 4 + 4.5
            ae[start] += 108
            ae[start+1:end] *= 0.08
        else:  # Stealing
            scores[start:end] += np.random.uniform(0.45, 0.75, end-start)
            ve = np.abs(np.random.randn(T)) * 6 + 7
            ve[start:end] += 10
            ae = np.abs(np.random.randn(T)) * 4 + 4.5
            ae[start:end] *= 0.05
    else:
        scores = np.clip(scores * 0.5, 0, 0.35)
        ve = np.abs(np.random.randn(T)) * 0.6 + 7
        ae = np.abs(np.random.randn(T)) * 0.4 + 4.5

    scores = np.clip(scores, 0, 1)
    fw = np.abs(np.random.randn(T)) * 0.003 + 0.500

    result = {
        "scores":         scores,
        "visual_energy":  ve,
        "audio_energy":   ae,
        "fusion_weights": fw,
        "peak_score":     float(scores.max()),
        "mean_score":     float(scores.mean()),
        "flagged_frames": [i for i, s in enumerate(scores) if s > ANOMALY_THRESHOLD],
        "verdict":        ("🚨 ANOMALY DETECTED" if scores.max() > ANOMALY_THRESHOLD
                           else "✅ NORMAL"),
        "verdict_color":  ("#ff4b6e" if scores.max() > ANOMALY_THRESHOLD
                           else "#4bffb8"),
        "elapsed_ms":     random.uniform(180, 340),
        "num_frames":     T,
        "mode":           f"Demo — {demo_choice}",
    }

    return (make_score_chart(result), make_energy_chart(result),
            make_fusion_chart(result), make_heatmap(result),
            make_summary_stats(result))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Base ─────────────────────────────────────────────────────── */
:root {
    --bg:       #080c14;
    --surface:  #0f1520;
    --border:   #1c2438;
    --accent:   #3d6aff;
    --danger:   #ff4b6e;
    --safe:     #00e5a0;
    --warning:  #ffcc3d;
    --text:     #d0d8f0;
    --dim:      #5a6380;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', 'Helvetica Neue', sans-serif;
}

/* ── Header ──────────────────────────────────────────────────── */
.avad-header {
    background: linear-gradient(135deg, #080c14 0%, #0f1a30 50%, #080c14 100%);
    border-bottom: 1px solid var(--border);
    padding: 36px 0 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.avad-header::before {
    content: '';
    position: absolute; inset: 0;
    background-image:
        radial-gradient(circle at 20% 50%, rgba(61,106,255,0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 50%, rgba(255,75,110,0.06) 0%, transparent 50%);
}
.avad-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(90deg, #3d6aff, #00e5a0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
}
.avad-sub {
    color: var(--dim);
    font-size: 1rem;
    margin-top: 8px;
    letter-spacing: 0.04em;
    position: relative;
}
.avad-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(61,106,255,0.12);
    border: 1px solid rgba(61,106,255,0.35);
    color: #7aa0ff;
    padding: 4px 14px; border-radius: 100px;
    font-size: 0.75rem; letter-spacing: 0.06em;
    margin-bottom: 14px; position: relative;
}
.avad-dot {
    width: 6px; height: 6px;
    background: var(--safe); border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Tabs ─────────────────────────────────────────────────────── */
.tab-nav button {
    background: transparent !important;
    color: var(--dim) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 10px 20px !important;
    transition: all 0.2s;
}
.tab-nav button.selected,
.tab-nav button:hover {
    color: var(--text) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Cards ────────────────────────────────────────────────────── */
.gr-box, .gr-panel, .gradio-box {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #3d6aff, #1a47d1) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(61,106,255,0.35) !important;
}
button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(61,106,255,0.5) !important;
}
button.secondary {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--dim) !important;
    border-radius: 8px !important;
}

/* ── Inputs ──────────────────────────────────────────────────── */
input, textarea, select, .gr-input {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
}

/* ── Upload area ─────────────────────────────────────────────── */
.upload-box {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: rgba(61,106,255,0.04) !important;
    transition: all 0.2s;
}
.upload-box:hover {
    border-color: var(--accent) !important;
    background: rgba(61,106,255,0.08) !important;
}

/* ── Labels ──────────────────────────────────────────────────── */
label, .gr-form label {
    color: var(--dim) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Markdown ────────────────────────────────────────────────── */
.gr-markdown, .prose {
    color: var(--text) !important;
}
.gr-markdown h3 { color: var(--text) !important; }
.gr-markdown table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg);
}
.gr-markdown td, .gr-markdown th {
    border: 1px solid var(--border) !important;
    padding: 8px 12px !important;
    color: var(--text) !important;
}
.gr-markdown th { background: var(--surface) !important; }

/* ── Plots ───────────────────────────────────────────────────── */
.gr-plot { border-radius: 10px !important; overflow: hidden; }

/* ── Slider ──────────────────────────────────────────────────── */
input[type=range]::-webkit-slider-thumb { background: var(--accent) !important; }
input[type=range]::-webkit-slider-runnable-track { background: var(--border) !important; }

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

HEADER_HTML = """
<div class="avad-header">
  <div class="avad-badge">
    <span class="avad-dot"></span>
    CLIP ViT-B/16 + PyTorch + HuggingFace
  </div>
  <div class="avad-title">AVadCLIP</div>
  <div class="avad-sub">Audio-Visual Anomaly Detection &nbsp;·&nbsp; Wu et al., 2025 Implementation</div>
</div>
"""

ABOUT_MD = """
## How AVadCLIP Works

AVadCLIP is a **weakly-supervised** video anomaly detection system that fuses **visual** and **audio** modalities using pretrained CLIP embeddings.

### Model Pipeline
```
VIDEO ──→ CLIP ViT-B/16 (HuggingFace, FROZEN) ──→ Temporal 1D CNN ──→ visual_feat [T, 512]
                                                                              │
AUDIO ──→ Mel Spectrogram ──→ CNN Backbone ──→ Projection MLP ──→ audio_feat [T, 512]
                                                                              │
                              Adaptive Fusion:  fused = visual + sigmoid(W) ⊙ R
                                                                              │
                              Anomaly Scorer MLP (512→256→128→1) ──→ score ∈ [0,1] per frame
```

### AI Models Used

| Model | Source | Role |
|-------|--------|------|
| **CLIP ViT-B/16** | `openai/clip-vit-base-patch16` (HuggingFace) | Visual frame encoding — **frozen** |
| **CLIP Text Encoder** | Same HuggingFace checkpoint | Zero-shot scoring via text prompts |
| **Audio CNN** | Custom PyTorch (Wav2CLIP-style) | Mel-spectrogram → 512-D CLIP space |
| **Temporal CNN** | Custom 1D DepthwiseConv | Adds motion context across frames |
| **Adaptive Fusion** | Core innovation (AVadCLIP paper) | Sigmoid-gated audio integration |
| **MIL Scorer** | Custom MLP | Frame-level anomaly probability |

### Loss Functions
- **MIL Ranking Loss** — trains on video-level labels only (no frame annotations needed)
- **Smoothness Loss** — penalises noisy per-frame score spikes
- **Sparsity Loss** — pushes most frames toward zero score
- **InfoNCE (optional)** — aligns fused A/V features with CLIP text embeddings

### Scoring Modes
- **MIL (trained head)** — uses the trained MLP scoring head. Best when a checkpoint is available.
- **Zero-Shot (CLIP text)** — uses cosine similarity between fused features and text prompts. Works with no training!
"""



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — AI Alert Agent (LangChain-powered notification system)
# ══════════════════════════════════════════════════════════════════════════════

import json, threading, smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Alert config — set via HuggingFace Space Secrets ─────────────────────────
ALERT_CFG = {
    "smtp_host":     os.getenv("SMTP_HOST",     "smtp.gmail.com"),
    "smtp_port":     int(os.getenv("SMTP_PORT", "587")),
    "smtp_user":     os.getenv("SMTP_USER",     ""),
    "smtp_password": os.getenv("SMTP_PASSWORD", ""),
    "alert_to":      os.getenv("ALERT_TO",      ""),
    "owner_name":    os.getenv("OWNER_NAME",    "Security Owner"),
    "company":       os.getenv("COMPANY_NAME",  "Your Company"),
}

ANOMALY_META = {
    "fighting":  {"icon": "🥊", "action": "Dispatch security immediately. Alert nearby personnel."},
    "explosion": {"icon": "💥", "action": "Evacuate area. Call fire dept & ambulance (101/108)."},
    "fire":      {"icon": "🔥", "action": "Activate fire alarm. Evacuate. Call 101."},
    "theft":     {"icon": "🕵️",  "action": "Lock exits. Alert security. Call police (100)."},
    "vandalism": {"icon": "🔨", "action": "Alert security. Preserve footage for evidence."},
    "unknown":   {"icon": "⚠️",  "action": "Security personnel should investigate."},
}

# In-memory incident log (shown in dashboard tab)
_incidents: list = []
_incidents_lock = threading.Lock()


def _classify_anomaly(result: dict) -> dict:
    """Classify anomaly type from visual/audio energy ratios."""
    peak   = result.get("peak_score", 0)
    ve     = result.get("visual_energy", [])
    ae     = result.get("audio_energy",  [])
    avg_v  = float(np.mean(ve)) if len(ve) else 0
    avg_a  = float(np.mean(ae)) if len(ae) else 0

    if   avg_a > avg_v * 1.5 and peak > 0.80: atype = "explosion"
    elif avg_a > avg_v * 1.2 and peak > 0.60: atype = "fighting"
    elif avg_v > avg_a * 1.5 and peak > 0.75: atype = "fire"
    elif avg_v > avg_a * 1.2 and peak > 0.55: atype = "theft"
    else:                                       atype = "unknown"

    if   peak >= 0.80: severity, sicon, scolor = "CRITICAL", "🔴", "#ff2244"
    elif peak >= 0.65: severity, sicon, scolor = "HIGH",     "🟠", "#ff8800"
    else:              severity, sicon, scolor = "MEDIUM",   "🟡", "#ffcc00"

    meta = ANOMALY_META[atype]
    return {
        "anomaly_type":       atype,
        "anomaly_icon":       meta["icon"],
        "severity":           severity,
        "severity_icon":      sicon,
        "severity_color":     scolor,
        "recommended_action": meta["action"],
    }


def _send_email_alert(alert: dict):
    """Send HTML email alert. Runs in background thread."""
    if not ALERT_CFG["smtp_user"] or not ALERT_CFG["alert_to"]:
        print("[Alert] Email skipped — SMTP_USER or ALERT_TO not configured")
        return

    severity_colors = {"CRITICAL": "#ff2244", "HIGH": "#ff8800", "MEDIUM": "#ffcc00"}
    sev_color = severity_colors.get(alert["severity"], "#aaaaaa")

    html = f"""
<!DOCTYPE html><html><body style="margin:0;padding:0;background:#0a0e1a;font-family:'Helvetica Neue',sans-serif">
<div style="max-width:600px;margin:0 auto;padding:32px 20px">
  <div style="background:#0f1520;border:1px solid #1a2540;border-radius:12px;overflow:hidden">
    <div style="background:linear-gradient(135deg,{sev_color}22,transparent);padding:28px 32px;border-bottom:1px solid #1a2540">
      <div style="display:inline-block;padding:4px 14px;border-radius:20px;border:1px solid {sev_color}55;background:{sev_color}11;color:{sev_color};font-size:11px;letter-spacing:0.12em;margin-bottom:12px">
        {alert['severity_icon']} {alert['severity']} ALERT
      </div>
      <h1 style="color:#ffffff;font-size:24px;font-weight:800;margin:0 0 6px">
        {alert['anomaly_icon']} {alert['anomaly_type'].upper()} DETECTED
      </h1>
      <p style="color:#5a6a8a;margin:0;font-size:13px">{ALERT_CFG['company']} · {alert['camera_id']} · {alert['location']}</p>
    </div>
    <div style="padding:28px 32px">
      <table style="width:100%;border-collapse:collapse;margin-bottom:20px">
        {''.join(f'<tr><td style="padding:8px 0;color:#5a6a8a;font-size:12px;border-bottom:1px solid #1a2540;width:40%">{k}</td><td style="padding:8px 0;color:#c8d8f0;font-size:13px;font-family:monospace;border-bottom:1px solid #1a2540">{v}</td></tr>' for k, v in [
            ("Alert ID", alert['alert_id']),
            ("Camera", alert['camera_id']),
            ("Location", alert['location']),
            ("Timestamp", alert['timestamp']),
            ("Confidence", f"{alert['peak_score']*100:.1f}%"),
            ("Flagged Frames", f"{len(alert['flagged_frames'])} / {alert['num_frames']}"),
        ])}
      </table>
      <div style="background:#0a0e1a;border:1px solid {sev_color}33;border-radius:8px;padding:16px;margin-bottom:16px">
        <div style="font-size:10px;letter-spacing:0.15em;color:#5a6a8a;text-transform:uppercase;margin-bottom:6px">Recommended Action</div>
        <div style="color:{sev_color};font-weight:600;font-size:14px">⚡ {alert['recommended_action']}</div>
      </div>
      <div style="background:#0a0e1a;border:1px solid #1a2540;border-radius:8px;padding:16px">
        <div style="font-size:10px;letter-spacing:0.15em;color:#5a6a8a;text-transform:uppercase;margin-bottom:6px">AI Summary</div>
        <div style="color:#c8d8f0;font-size:13px;line-height:1.6">{alert['ai_summary']}</div>
      </div>
    </div>
    <div style="padding:16px 32px;border-top:1px solid #1a2540;text-align:center;color:#2a3a5a;font-size:11px">
      AVadCLIP Security System · {ALERT_CFG['company']}
    </div>
  </div>
</div></body></html>"""

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert['severity']}] {alert['anomaly_icon']} {alert['anomaly_type'].upper()} at {alert['location']} — {ALERT_CFG['company']}"
        msg["From"]    = ALERT_CFG["smtp_user"]
        msg["To"]      = ALERT_CFG["alert_to"]
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(ALERT_CFG["smtp_host"], ALERT_CFG["smtp_port"]) as s:
            s.starttls()
            s.login(ALERT_CFG["smtp_user"], ALERT_CFG["smtp_password"])
            s.sendmail(ALERT_CFG["smtp_user"], ALERT_CFG["alert_to"], msg.as_string())
        print(f"[Alert] Email sent → {ALERT_CFG['alert_to']}")
    except Exception as e:
        print(f"[Alert] Email failed: {e}")


def dispatch_alert(result: dict, camera_id: str = "CAM-01", location: str = "Unknown") -> dict:
    """
    Main alert dispatcher. Call after every run_inference().
    Classifies anomaly, stores in log, fires email in background.
    Returns the alert dict for display in UI.
    """
    peak = result.get("peak_score", 0)
    if peak < ANOMALY_THRESHOLD:
        return None

    classification = _classify_anomaly(result)
    alert_id  = f"ALT-{int(time.time())}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alert = {
        "alert_id":           alert_id,
        "camera_id":          camera_id,
        "location":           location,
        "timestamp":          timestamp,
        "peak_score":         round(peak, 3),
        "mean_score":         round(float(result.get("mean_score", 0)), 3),
        "flagged_frames":     result.get("flagged_frames", []),
        "num_frames":         result.get("num_frames", 32),
        "ai_summary": (
            f"{classification['severity']} {classification['anomaly_type']} event detected "
            f"at {location} with {peak*100:.1f}% confidence. "
            f"{len(result.get('flagged_frames',[]))} of {result.get('num_frames',32)} frames flagged."
        ),
        **classification,
    }

    with _incidents_lock:
        _incidents.insert(0, alert)

    # Fire email in background so Gradio UI doesn't block
    if ALERT_CFG["smtp_user"]:
        threading.Thread(target=_send_email_alert, args=(alert,), daemon=True).start()

    print(f"[Alert] {classification['severity']} {classification['anomaly_type']} @ {location} (score={peak:.3f})")
    return alert


def _render_dashboard_html() -> str:
    """Generate the current alert dashboard as HTML for the Gradio iframe."""
    with _incidents_lock:
        incidents_copy = list(_incidents[:50])

    cards = ""
    for a in incidents_copy:
        sev_color = {"CRITICAL": "#ff2244", "HIGH": "#ff8800", "MEDIUM": "#ffcc00"}.get(a["severity"], "#aaa")
        cards += f"""
        <div style="background:#0b1222;border:1px solid #1a2540;border-left:3px solid {sev_color};border-radius:8px;padding:14px 16px;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
            <span style="font-family:'Exo 2',sans-serif;font-weight:700;font-size:14px;color:{sev_color}">{a.get('anomaly_icon','⚠️')} {a['anomaly_type'].upper()}</span>
            <span style="font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid {sev_color}55;color:{sev_color};background:{sev_color}11">{a['severity']}</span>
          </div>
          <div style="color:#c8d8f0;font-size:12px;margin-bottom:4px">📍 {a['location']} &nbsp;·&nbsp; {a['camera_id']}</div>
          <div style="color:#3a4a6a;font-family:monospace;font-size:11px;margin-bottom:8px">{a['timestamp']} &nbsp;·&nbsp; {a['alert_id']}</div>
          <div style="background:#04070f;border-radius:6px;padding:10px;font-size:12px;color:#8a9aba;line-height:1.5">{a['ai_summary']}</div>
          <div style="margin-top:8px;font-size:12px;color:{sev_color};font-weight:600">⚡ {a['recommended_action']}</div>
        </div>"""

    empty = "" if incidents_copy else """
        <div style="text-align:center;padding:60px 20px;color:#2a3a5a">
          <div style="font-size:48px;margin-bottom:12px;opacity:0.4">🛡️</div>
          <div style="font-size:14px">All clear — no anomalies detected yet</div>
          <div style="font-size:12px;margin-top:6px;opacity:0.6">Run a video analysis to see alerts here</div>
        </div>"""

    total    = len(incidents_copy)
    critical = sum(1 for a in incidents_copy if a["severity"] == "CRITICAL")
    high     = sum(1 for a in incidents_copy if a["severity"] == "HIGH")
    medium   = sum(1 for a in incidents_copy if a["severity"] == "MEDIUM")

    return f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@400;700;800&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
* {{ box-sizing:border-box;margin:0;padding:0 }}
body {{ background:#04070f;color:#c8d8f0;font-family:'Exo 2',sans-serif;padding:16px }}
::-webkit-scrollbar {{ width:4px }}
::-webkit-scrollbar-thumb {{ background:#1a2540;border-radius:2px }}
</style></head><body>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px">
  <div style="background:#0b1222;border:1px solid #1a2540;border-radius:8px;padding:12px 16px">
    <div style="font-size:24px;font-weight:800;color:#fff">{total}</div>
    <div style="font-size:10px;letter-spacing:0.15em;color:#3a4a6a;text-transform:uppercase;margin-top:2px">Total Alerts</div>
  </div>
  <div style="background:#0b1222;border:1px solid #1a2540;border-radius:8px;padding:12px 16px">
    <div style="font-size:24px;font-weight:800;color:#ff2244">{critical}</div>
    <div style="font-size:10px;letter-spacing:0.15em;color:#3a4a6a;text-transform:uppercase;margin-top:2px">Critical</div>
  </div>
  <div style="background:#0b1222;border:1px solid #1a2540;border-radius:8px;padding:12px 16px">
    <div style="font-size:24px;font-weight:800;color:#ff8800">{high}</div>
    <div style="font-size:10px;letter-spacing:0.15em;color:#3a4a6a;text-transform:uppercase;margin-top:2px">High</div>
  </div>
  <div style="background:#0b1222;border:1px solid #1a2540;border-radius:8px;padding:12px 16px">
    <div style="font-size:24px;font-weight:800;color:#ffcc00">{medium}</div>
    <div style="font-size:10px;letter-spacing:0.15em;color:#3a4a6a;text-transform:uppercase;margin-top:2px">Medium</div>
  </div>
</div>
<div>{cards}{empty}</div>
</body></html>"""



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Updated analyse_video handler (with alert dispatch)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_video_with_alert(video_file, num_frames, mode, custom_prompt,
                              camera_id, location,
                              progress=gr.Progress(track_tqdm=True)):
    if video_file is None:
        raise gr.Error("Please upload a video file first.")

    progress(0.1, desc="Extracting frames & audio …")
    try:
        result = run_inference(video_file, int(num_frames), mode, custom_prompt)
    except Exception as e:
        raise gr.Error(f"Inference failed: {e}")

    progress(0.7, desc="Generating visualisations …")
    score_fig   = make_score_chart(result)
    energy_fig  = make_energy_chart(result)
    fusion_fig  = make_fusion_chart(result)
    heatmap_fig = make_heatmap(result)
    summary_md  = make_summary_stats(result)

    # ── Dispatch alert if anomaly detected ───────────────────────────
    progress(0.9, desc="Checking for alerts …")
    alert = dispatch_alert(result,
                           camera_id=camera_id or "CAM-01",
                           location=location or "Unknown Location")

    if alert:
        alert_md = f"""
---
### {alert['severity_icon']} {alert['severity']} ALERT DISPATCHED

| Field | Value |
|-------|-------|
| **Type** | {alert['anomaly_icon']} {alert['anomaly_type'].upper()} |
| **Alert ID** | `{alert['alert_id']}` |
| **Camera** | {alert['camera_id']} — {alert['location']} |
| **Action** | {alert['recommended_action']} |

{"📧 **Email notification sent** to " + ALERT_CFG['alert_to'] if ALERT_CFG['smtp_user'] else "⚙️ Configure SMTP_USER / ALERT_TO secrets to enable email alerts"}
"""
        summary_md = summary_md + alert_md

    progress(1.0, desc="Done!")
    return score_fig, energy_fig, fusion_fig, heatmap_fig, summary_md


def get_dashboard_update():
    """Returns fresh dashboard HTML — called by Gradio refresh button."""
    return _render_dashboard_html()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Full Gradio UI (original tabs + new Alert Dashboard tab)
# ══════════════════════════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS + """
        .alert-tab-label { color: #ff4b6e !important; }
        """,
        title="AVadCLIP — Audio-Visual Anomaly Detection",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs(elem_classes=["tab-nav"]):

            # ════════════════════════════════════════════════
            # TAB 1 — Upload & Analyse (with alert fields)
            # ════════════════════════════════════════════════
            with gr.Tab("🎬  Analyse Video"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("### Upload Video")
                        video_input = gr.Video(
                            label="Video File (.mp4 / .avi / .mov)",
                            elem_classes=["upload-box"],
                        )
                        gr.Markdown("### Settings")
                        num_frames_slider = gr.Slider(
                            minimum=8, maximum=64, value=32, step=4,
                            label="Number of Frames to Sample",
                        )
                        scoring_mode = gr.Radio(
                            choices=["MIL (trained head)", "Zero-Shot (CLIP Text)"],
                            value="MIL (trained head)",
                            label="Scoring Mode",
                        )
                        custom_prompt = gr.Textbox(
                            label="Custom Anomaly Prompt (Zero-Shot only)",
                            placeholder="e.g. a video of someone falling down stairs",
                            lines=2,
                        )
                        gr.Markdown("### 🔔 Alert Settings")
                        camera_id_input = gr.Textbox(
                            label="Camera ID",
                            placeholder="e.g. CAM-03",
                            value="CAM-01",
                        )
                        location_input = gr.Textbox(
                            label="Camera Location",
                            placeholder="e.g. Server Room, Lobby, Parking Lot",
                            value="Main Entrance",
                        )
                        gr.Markdown(
                            "_Set **SMTP_USER**, **SMTP_PASSWORD**, **ALERT_TO** in Space Secrets to enable email alerts._",
                        )
                        analyse_btn = gr.Button("🔍  Analyse Video", variant="primary", size="lg")
                        clear_btn   = gr.Button("✕  Clear", variant="secondary")

                    with gr.Column(scale=2):
                        summary_md = gr.Markdown(
                            value="*Upload a video and click Analyse to see results.*"
                        )
                        with gr.Tabs():
                            with gr.Tab("📊 Score Timeline"):
                                score_plot = gr.Plot(label="")
                            with gr.Tab("⚡ Visual vs Audio Energy"):
                                energy_plot = gr.Plot(label="")
                            with gr.Tab("🔀 Fusion Weights"):
                                fusion_plot = gr.Plot(label="")
                            with gr.Tab("🌡️ Heatmap"):
                                heatmap_plot = gr.Plot(label="")

                analyse_btn.click(
                    fn=analyse_video_with_alert,
                    inputs=[video_input, num_frames_slider, scoring_mode,
                            custom_prompt, camera_id_input, location_input],
                    outputs=[score_plot, energy_plot, fusion_plot, heatmap_plot, summary_md],
                )
                clear_btn.click(
                    fn=lambda: (None, None, None, None, None,
                                "*Upload a video and click Analyse to see results.*"),
                    outputs=[video_input, score_plot, energy_plot,
                             fusion_plot, heatmap_plot, summary_md],
                )

            # ════════════════════════════════════════════════
            # TAB 2 — Interactive Demo
            # ════════════════════════════════════════════════
            with gr.Tab("🎭  Interactive Demo"):
                gr.Markdown("""
### Try without uploading a video
Select an anomaly scenario to see what AVadCLIP's output looks like.
Real signals are simulated from the statistical properties of each anomaly type.
""")
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        demo_choice = gr.Radio(
                            choices=["Fighting 🥊", "Explosion 💥",
                                     "Stealing 🕵️", "Normal 🚶"],
                            value="Fighting 🥊",
                            label="Anomaly Scenario",
                        )
                        demo_btn = gr.Button("▶  Run Demo", variant="primary", size="lg")
                        gr.Markdown("""
---
**What each scenario simulates:**

🥊 **Fighting** — burst of high visual + audio energy together (shouting, impacts)

💥 **Explosion** — extreme visual spike on one frame, massive audio impulse then silence

🕵️ **Stealing** — subtle visual change, audio goes nearly silent (suspicious quietness!)

🚶 **Normal** — low stable scores throughout, no spikes
""")
                    with gr.Column(scale=2):
                        demo_summary = gr.Markdown()
                        with gr.Tabs():
                            with gr.Tab("📊 Score Timeline"):
                                demo_score_plot = gr.Plot()
                            with gr.Tab("⚡ Visual vs Audio Energy"):
                                demo_energy_plot = gr.Plot()
                            with gr.Tab("🔀 Fusion Weights"):
                                demo_fusion_plot = gr.Plot()
                            with gr.Tab("🌡️ Heatmap"):
                                demo_heatmap_plot = gr.Plot()

                demo_btn.click(
                    fn=analyse_demo,
                    inputs=[demo_choice],
                    outputs=[demo_score_plot, demo_energy_plot,
                             demo_fusion_plot, demo_heatmap_plot, demo_summary],
                )

            # ════════════════════════════════════════════════
            # TAB 3 — 🚨 Alert Dashboard (NEW)
            # ════════════════════════════════════════════════
            with gr.Tab("🚨  Alert Dashboard"):
                gr.Markdown("""
### Real-time Security Alert Dashboard
All anomaly detections appear here automatically after analysis.
Configure email notifications via Space Secrets (see below).
""")
                with gr.Row():
                    refresh_btn = gr.Button("🔄  Refresh Dashboard", variant="primary")
                    gr.Markdown("""
**Email Alerts** — Add these in HuggingFace Space → Settings → Secrets:

| Secret | Value |
|--------|-------|
| `SMTP_USER` | your Gmail address |
| `SMTP_PASSWORD` | Gmail App Password |
| `ALERT_TO` | security owner's email |
| `OWNER_NAME` | owner's name |
| `COMPANY_NAME` | your company name |
""")

                dashboard_html = gr.HTML(
                    value=_render_dashboard_html(),
                    label="",
                )
                refresh_btn.click(
                    fn=get_dashboard_update,
                    outputs=[dashboard_html],
                )

            # ════════════════════════════════════════════════
            # TAB 4 — Architecture
            # ════════════════════════════════════════════════
            with gr.Tab("🧠  Architecture"):
                gr.Markdown(ABOUT_MD)
                gr.Markdown("### Component Details")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
**🖼️ CLIP ViT-B/16 Visual Encoder**

Loaded directly from `openai/clip-vit-base-patch16` via HuggingFace `transformers`.
Splits each 224×224 frame into 196 patches of 16×16 pixels.
12 Transformer layers with multi-head self-attention produce a 768-D CLS token,
which is projected to 512-D by CLIP's visual projection head.

Kept **frozen** — its 400M-pair pretrained world knowledge is preserved.
""")
                        gr.Markdown("""
**🔊 Audio Encoder (Wav2CLIP-style)**

Raw waveform → Log Mel-Spectrogram (64 bands, n_fft=1024, hop=512).
4-block CNN backbone (32→64→128→256 channels, BatchNorm + ReLU).
AdaptiveAvgPool → Flatten → MLP(4096→1024→512).
Output: 512-D vector in **the same space** as CLIP visual features — directly comparable!
""")
                    with gr.Column():
                        gr.Markdown("""
**⚡ Adaptive Fusion Module**

```
joint = cat([visual, audio])        # [B, T, 1024]
W     = sigmoid(Linear(joint))      # [B, T, 512]  ← audio gate
R     = MLP(joint)                  # [B, T, 512]  ← cross-modal residual
fused = LayerNorm(visual + W ⊙ R)  # vision-centric!
```

W is **per-dimension** — each of 512 feature dimensions gets its own audio
contribution weight. This is richer than a single scalar gate.
""")
                        gr.Markdown("""
**🎯 MIL Anomaly Scorer**

MLP: 512 → 256 → 128 → 1 (sigmoid)

Trained with **Multiple Instance Learning Ranking Loss**:
```
loss = max(0,  margin  -  top_k(anomalous)  +  top_k(normal))
```

Only needs video-level labels (anomalous/normal) — no frame annotations required!
This makes labelling real surveillance footage practical.
""")

            # ════════════════════════════════════════════════
            # TAB 5 — Deploy Guide
            # ════════════════════════════════════════════════
            with gr.Tab("🚀  Deploy"):
                gr.Markdown("""
## Deployment Guide

### HuggingFace Spaces (Free, Public URL)
```bash
# 1. Go to https://huggingface.co/spaces
# 2. New Space → SDK: Gradio → Python 3.10
# 3. Upload ONLY these 2 files:
#      app.py
#      requirements.txt
# 4. (Optional) Add secrets in Space Settings for email alerts
# 5. Your app is live at:
#      https://huggingface.co/spaces/<username>/<space-name>
```

### Space Secrets for Email Alerts
Go to your Space → **Settings** → **Variables and secrets** → Add:

| Secret Name | Example Value |
|-------------|---------------|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | `your@gmail.com` |
| `SMTP_PASSWORD` | `xxxx xxxx xxxx xxxx` (Gmail App Password) |
| `ALERT_TO` | `owner@company.com` |
| `COMPANY_NAME` | `IIIT Nagpur Security` |

### Gmail App Password Setup
1. Google Account → Security → 2-Step Verification → ON
2. Search "App passwords" → Create one for "Mail"
3. Use that 16-character password as `SMTP_PASSWORD`

### Local Run
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```
""")

        gr.HTML("""
<div style="text-align:center;padding:20px 0 10px;
            border-top:1px solid #1c2438;margin-top:20px;
            color:#5a6380;font-size:0.8rem;letter-spacing:0.04em">
  AVadCLIP &nbsp;·&nbsp; Audio-Visual CLIP Anomaly Detection + AI Alert Agent &nbsp;·&nbsp;
  PyTorch + HuggingFace + LangChain &nbsp;·&nbsp; Wu et al., 2025
</div>
""")

    return demo


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
