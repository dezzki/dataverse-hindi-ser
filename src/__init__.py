"""DataVerse — Hindi Speech Emotion Recognition (4-class, macro-F1) pipeline.

Modules
-------
config      ExperimentConfig dataclass + YAML IO (no heavy deps)
data        CSV readers + audio loading (librosa/soundfile)
splits      stratified K-fold generation / persistence
augment     waveform & spectrogram augmentation + mixup
dataset     torch ``Dataset`` yielding (waveform[C,T], label)
features    classical feature extraction (eGeMAPS / MFCC stats)
modeling    SSL backbone + pooling head factory (wav2vec2/wavlm/hubert/whisper)
train_ssl   fine-tuning CLI (manual loop, AMP, grad-accum, early stop)
predict     inference CLI (OOF + test probs, optional TTA)
eval        metrics, confusion matrix, prior/bias calibration search
ensemble    OOF-weighted blending + temperature scaling + calibration
classic     classical baseline (eGeMAPS+MFCC -> SVM / LightGBM)
submit      submission.csv writer + hard schema validation
kaggle_setup Kaggle notebook helper (data root detection / config override)
"""

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "splits",
    "augment",
    "dataset",
    "features",
    "modeling",
    "train_ssl",
    "predict",
    "eval",
    "ensemble",
    "classic",
    "submit",
    "kaggle_setup",
]
