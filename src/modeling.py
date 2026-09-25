"""Model factory: pretrained speech SSL backbone + pooling head.

Supported backbones (auto-detected from the checkpoint config):

* ``wav2vec2`` / ``wavlm`` / ``hubert`` / ``sew`` / ``data2vec-audio`` — raw
  waveform ``input_values`` -> ``last_hidden_state`` (B, T, D)
* ``whisper`` — log-mel ``input_features`` -> ``encoder`` last hidden state.
  The decoder is never instantiated at inference time (only ``model.encoder``
  is kept in the wrapper).

The classification head is shared by every family:
``pooling(meanstd -> 2D | attention -> D)`` -> ``Linear(D', 256)`` -> GELU ->
``Dropout`` -> ``Linear(256, num_labels)``.

``superb/wav2vec2-base-superb-er`` and friends load transparently through the
generic path (their classification head weights are simply ignored).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import ExperimentConfig

LOG = logging.getLogger(__name__)

__all__ = ["SERModel", "build_model", "prepare_inputs", "load_state_dict_flexible"]

_SSL_MODEL_TYPES = {"wav2vec2", "wavlm", "hubert", "sew", "sew_d", "data2vec-audio", "unispeech", "unispeech-sat", "mctct"}
_WHISPER_MODEL_TYPES = {"whisper"}


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #
class MeanStdPooling(nn.Module):
    """Concatenate masked mean and (unbiased) std over the time axis."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            return torch.cat([hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)], dim=-1)
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)  # (B, T, 1)
        denom = mask.sum(dim=1).clamp(min=self.eps)
        mean = (hidden * mask).sum(dim=1) / denom
        var = (((hidden - mean.unsqueeze(1)) * mask) ** 2).sum(dim=1) / denom
        return torch.cat([mean, torch.sqrt(var.clamp(min=self.eps))], dim=-1)


class AttentionPooling(nn.Module):
    """Single-head additive attention pooling with mask support."""

    def __init__(self, hidden_size: int, attn_dim: Optional[int] = None) -> None:
        super().__init__()
        attn_dim = int(attn_dim or hidden_size)
        self.score = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.score(hidden).squeeze(-1)  # (B, T)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.to(torch.bool) == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)  # all-masked safety
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class SERModel(nn.Module):
    """SSL backbone + pooling + MLP classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int = 4,
        pooling: str = "meanstd",
        head_hidden: int = 256,
        dropout: float = 0.3,
        family: str = "ssl",
        spec_augment: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.family = family
        self.num_labels = int(num_labels)
        self.pooling = pooling

        if pooling in ("meanstd",):
            self.pool = MeanStdPooling()
            pooled_dim = hidden_size * 2
        elif pooling in ("attention", "meanstd_attn"):
            self.pool = AttentionPooling(hidden_size)
            pooled_dim = hidden_size
        elif pooling == "mean":
            self.pool = None
            pooled_dim = hidden_size
        else:
            raise ValueError(f"Unknown pooling {pooling!r}")

        self.head = nn.Sequential(
            nn.Linear(pooled_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), self.num_labels),
        )
        self._enable_specaug(spec_augment)

    # ------------------------------------------------------------------ #
    def _enable_specaug(self, spec_augment: Optional[Dict[str, Any]]) -> None:
        """Turn on HF's built-in SpecAugment on the encoder hidden states."""
        if not spec_augment or not spec_augment.get("enabled", False):
            return
        cfg = getattr(self.backbone, "config", None)
        if cfg is None:
            return
        try:
            cfg.apply_spec_augment = True
            cfg.mask_time_prob = float(spec_augment.get("mask_time_prob", 0.05))
            cfg.mask_time_length = int(spec_augment.get("time_mask_width", 10))
            cfg.mask_feature_prob = float(spec_augment.get("mask_feature_prob", 0.0))
            cfg.mask_feature_length = int(spec_augment.get("freq_mask_width", 10))
            LOG.info(
                "Encoder SpecAugment enabled (p_time=%.3f, p_feat=%.3f)",
                cfg.mask_time_prob,
                cfg.mask_feature_prob,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug("Could not enable encoder SpecAugment: %s", exc)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Return logits ``(B, num_labels)``."""
        if self.family == "whisper":
            feats = input_features if input_features is not None else input_values
            if feats is None:
                raise ValueError("Whisper family requires input_features")
            enc = self.backbone.encoder(
                feats.to(next(self.backbone.encoder.parameters()).dtype),
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = enc.last_hidden_state
            mask = None
            if attention_mask is not None and attention_mask.shape[-1] == hidden.shape[1]:
                mask = attention_mask
        else:
            wave = input_values if input_values is not None else input_features
            if wave is None:
                raise ValueError("SSL family requires input_values")
            if wave.dim() == 3 and wave.size(1) == 1:
                wave = wave.squeeze(1)  # (B, 1, T) -> (B, T)
            if wave.dim() == 3 and wave.size(1) > 1:
                wave = wave.mean(dim=1)  # stereo safeguard
            out = self.backbone(input_values=wave, attention_mask=attention_mask, return_dict=True)
            hidden = out.last_hidden_state
            mask = self._frame_mask(attention_mask, hidden.shape[1])

        if self.pooling == "mean":
            pooled = hidden.mean(dim=1)
        else:
            pooled = self.pool(hidden, mask)
        return self.head(pooled.to(self.head[0].weight.dtype))

    def _frame_mask(self, attention_mask: Optional[torch.Tensor], n_frames: int) -> Optional[torch.Tensor]:
        """Convert a sample-level attention mask to encoder-frame level.

        ``Wav2Vec2FeatureExtractor`` returns an attention mask in *samples*
        (e.g. 96000) while ``last_hidden_state`` has ~299 frames; pooling must
        use a mask of the latter length.
        """
        if attention_mask is None:
            return None
        if attention_mask.shape[-1] == n_frames:
            return attention_mask
        downsampler = getattr(self.backbone, "_get_feature_vector_attention_mask", None)
        if downsampler is not None:
            try:
                return downsampler(n_frames, attention_mask)
            except Exception as exc:  # pragma: no cover - architecture dependent
                LOG.debug("Attention-mask downsample failed (%s); falling back to nearest slicing", exc)
        ratio = attention_mask.shape[-1] / max(n_frames, 1)
        idx = (
            torch.arange(n_frames, device=attention_mask.device, dtype=torch.float32) * ratio
        ).long().clamp(max=attention_mask.shape[-1] - 1)
        return attention_mask[:, idx]

    # ------------------------------------------------------------------ #
    def encoder_parameters(self) -> list:
        return list(self.backbone.parameters())

    def head_parameters(self) -> list:
        params = list(self.head.parameters())
        if isinstance(self.pool, nn.Module):
            params += list(self.pool.parameters())
        return params

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def _infer_family(model_type: str, name: str) -> str:
    mt = (model_type or "").lower()
    if mt in _WHISPER_MODEL_TYPES:
        return "whisper"
    if mt in _SSL_MODEL_TYPES:
        return "ssl"
    if "whisper" in name.lower():
        return "whisper"
    return "ssl"


def _load_backbone(cfg: ExperimentConfig, family: str) -> Tuple[nn.Module, int, Any]:
    """Load the pretrained backbone + feature extractor + hidden size."""
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoProcessor

    local = bool(getattr(cfg, "model_local_dir", ""))
    model_ref = cfg.model_local_dir or cfg.model_name
    kw = {"local_files_only": True} if local else {}
    hf_config = AutoConfig.from_pretrained(model_ref, **kw)
    model_type = getattr(hf_config, "model_type", "")
    hidden_size = int(
        getattr(hf_config, "hidden_size", None)
        or getattr(hf_config, "d_model", None)
        or getattr(hf_config, "n_embd", None)
        or 768
    )

    backbone = AutoModel.from_pretrained(model_ref, ignore_mismatched_sizes=True, **kw)
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False

    # Feature extractor: AudioProcessor (transformers v5) first, then the
    # classic AutoFeatureExtractor.
    fe = None
    for factory in (AutoProcessor, AutoFeatureExtractor):
        try:
            fe = factory.from_pretrained(model_ref, **kw)
            break
        except Exception as exc:  # pragma: no cover - depends on checkpoint
            LOG.debug("%s could not load a processor for %s: %s", factory.__name__, model_ref, exc)
    if fe is None:
        raise RuntimeError(f"No feature extractor/processor found for {model_ref!r}")

    if family == "whisper":
        fe = _configure_whisper_window(backbone, fe, cfg)
    LOG.info(
        "Loaded %s (%s, hidden=%d, family=%s) via %s (local=%s)",
        model_ref,
        model_type or "unknown",
        hidden_size,
        family,
        type(backbone).__name__,
        local,
    )
    return backbone, hidden_size, fe


def _inner_feature_extractor(fe: Any) -> Any:
    """Unwrap a processor (e.g. ``WhisperProcessor``) to its feature extractor.

    ``AutoProcessor`` returns a processor whose window attributes are *not*
    proxied to the inner feature extractor, so configuration must be applied to
    the wrapped object.
    """
    for attr in ("feature_extractor", "audio_processor", "image_processor"):
        inner = getattr(fe, attr, None)
        if inner is not None and hasattr(inner, "n_samples"):
            return inner
    return fe


def _configure_whisper_window(backbone: nn.Module, fe: Any, cfg: ExperimentConfig) -> Any:
    """Match Whisper's fixed 30 s window to ``cfg.max_seconds``.

    ``WhisperEncoder`` hard-requires exactly ``2 * config.max_source_positions``
    mel frames and adds learned positional embeddings of that length. To run
    6 s clips (instead of the full 30 s) we:

    1. trim ``encoder.embed_positions`` to the first ``max_seconds`` of the
       pretrained window (positions are shared across the beginning of the
       input, so the first seconds behave identically to full-window training),
    2. update ``config.max_source_positions`` accordingly,
    3. set ``feature_extractor.n_samples`` so the mel frames line up exactly.

    Falls back to the checkpoint's original window if anything looks unusual.
    """
    try:
        fe = _inner_feature_extractor(fe)
        hop = int(getattr(fe, "hop_length", 160) or 160)
        if int(getattr(fe, "sampling_rate", cfg.sample_rate)) != int(cfg.sample_rate):
            raise ValueError(
                f"Whisper requires sample_rate=16000, got {cfg.sample_rate}. "
                "Set sample_rate: 16000 in the config."
            )
        original_positions = int(getattr(backbone.config, "max_source_positions", 1500))
        target_frames = (int(cfg.max_samples) // hop) - ((int(cfg.max_samples) // hop) % 2)
        positions = max(1, min(target_frames // 2, original_positions))
        if positions < original_positions:
            emb = getattr(getattr(backbone, "encoder", None), "embed_positions", None)
            if emb is not None:
                new_emb = nn.Embedding(positions, emb.embedding_dim)
                with torch.no_grad():
                    new_emb.weight.copy_(emb.weight[:positions].detach())
                new_emb.weight.requires_grad_(False)
                backbone.encoder.embed_positions = new_emb  # type: ignore[union-attr]
                backbone.config.max_source_positions = positions
            if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "max_source_positions"):
                backbone.encoder.max_source_positions = positions
        frames = 2 * int(getattr(backbone.config, "max_source_positions", positions))
        n_samples = frames * hop
        for attr, value in (
            ("n_samples", n_samples),
            ("nb_max_frames", frames),
            ("chunk_length", n_samples / float(cfg.sample_rate)),
        ):
            if hasattr(fe, attr):
                setattr(fe, attr, value)
        LOG.info(
            "Whisper window -> %d mel frames (%.2fs, %d positional embeddings)",
            frames,
            n_samples / float(cfg.sample_rate),
            positions,
        )
    except Exception as exc:
        LOG.warning("Could not resize Whisper window (%s); using the checkpoint's default 30 s window.", exc)
    return fe

def _freeze_first_layers(model: nn.Module, n_layers: int) -> None:
    """Freeze the first ``n_layers`` transformer blocks of the encoder."""
    if n_layers <= 0:
        return
    encoder = getattr(model, "encoder", None) or getattr(getattr(model, "backbone", None), "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        LOG.warning("freeze_encoder_layers=%d requested but no encoder.layers found", n_layers)
        return
    frozen = 0
    for layer in list(layers)[: int(n_layers)]:
        for param in layer.parameters():
            param.requires_grad = False
        frozen += 1
    LOG.info("Froze the first %d encoder layer(s)", frozen)


def build_model(
    cfg: ExperimentConfig,
    num_labels: int = 4,
    device: Optional[str] = None,
) -> Tuple[SERModel, Any]:
    """Build ``(model, feature_extractor)`` for the checkpoint in ``cfg``.

    The returned model is in ``train()`` mode unless ``device`` is ``None`` and
    the caller wants to manage that; ``device`` (when given) moves it.
    """
    from transformers import AutoConfig

    local = bool(getattr(cfg, "model_local_dir", ""))
    hf_config = AutoConfig.from_pretrained(
        cfg.model_local_dir or cfg.model_name,
        **({"local_files_only": True} if local else {}),
    )
    family = cfg.model_family if cfg.model_family not in ("", "auto", None) else _infer_family(
        getattr(hf_config, "model_type", ""), cfg.model_name
    )
    backbone, hidden_size, fe = _load_backbone(cfg, family)

    if family == "whisper":
        # Keep only the encoder to save ~half the parameters (decoder unused).
        try:
            backbone.decoder = None  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive
            LOG.debug("Could not drop Whisper decoder (not fatal)")

    model = SERModel(
        backbone=backbone,
        hidden_size=hidden_size,
        num_labels=num_labels,
        pooling=cfg.pooling,
        head_hidden=cfg.head_hidden,
        dropout=cfg.dropout,
        family=family,
        spec_augment=cfg.aug.get("specaug") if isinstance(cfg.aug, dict) else None,
    )

    _freeze_first_layers(model, int(cfg.freeze_encoder_layers))

    if cfg.grad_ckpt:
        target = model.backbone
        try:
            target.gradient_checkpointing_enable()
            if hasattr(target, "enable_input_require_grads"):
                target.enable_input_require_grads()
            LOG.info("Gradient checkpointing enabled")
        except Exception as exc:  # pragma: no cover - depends on architecture
            LOG.warning("gradient_checkpointing_enable() failed: %s", exc)

    if device:
        model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = model.trainable_parameter_count()
    LOG.info(
        "Model ready: %.1fM params total, %.1fM trainable (freeze=%d, grad_ckpt=%s)",
        n_total / 1e6,
        n_train / 1e6,
        cfg.freeze_encoder_layers,
        cfg.grad_ckpt,
    )
    return model, fe


# --------------------------------------------------------------------------- #
# Input preparation / checkpoint IO
# --------------------------------------------------------------------------- #
@torch.no_grad()
def prepare_inputs(
    waves: torch.Tensor,
    feature_extractor: Any,
    sample_rate: int = 16_000,
) -> Dict[str, torch.Tensor]:
    """Run the HF feature extractor on a ``(B, C, T)`` batch.

    Returns the ``BatchFeature`` dict (``input_values`` for SSL backbones,
    ``input_features`` for Whisper), possibly with ``attention_mask``. The keys
    line up with :meth:`SERModel.forward`.
    """
    if waves.dim() == 3:
        waves = waves.squeeze(1) if waves.size(1) == 1 else waves.mean(dim=1)
    arrays = [w.detach().cpu().numpy().astype("float32") for w in waves]
    try:
        batch = feature_extractor(
            arrays, sampling_rate=int(sample_rate), return_tensors="pt", padding=True
        )
    except TypeError:  # very old/new FE signatures
        batch = feature_extractor(arrays, return_tensors="pt", padding=True)
    return {k: v for k, v in dict(batch).items() if isinstance(v, torch.Tensor)}


def load_state_dict_flexible(model: nn.Module, state: Dict[str, Any], strict: bool = False):
    """Load a checkpoint that may carry ``module.`` / ``model.`` prefixes."""
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if missing:
        LOG.warning("Missing keys when loading checkpoint: %s", list(missing)[:5])
    if unexpected:
        LOG.warning("Unexpected keys when loading checkpoint: %s", list(unexpected)[:5])
    return missing, unexpected
