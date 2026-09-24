# DataVerse — Detecting Emotions from Hindi Speech

Kaggle competition pipeline: **4-class Hindi speech emotion recognition**
(`angry`, `happy`, `neutral`, `sad`), metric = **macro F1**.

```
data/            train.csv, test.csv, sample_submission.csv, train/, test/  (wav)
src/             pipeline modules (see table below)
configs/         one YAML per model / baseline
runs/            training outputs (<tag>/fold{k}/best.pt, logs, ensembles)
oof/             OOF + test probabilities (<tag>/oof_probs_<tag>.csv, ...)
```

Two supported environments, one codebase — every path comes from the config:

| Env | `data_root` |
|---|---|
| local workstation (RTX 3050 6 GB) | `data/SRCASW-BhavVaani` (this repo's download) or `data` |
| Kaggle notebook | `/kaggle/input/<competition>` |

> `data_root` must be the directory that directly contains `train.csv`,
> `test.csv` and the `train/` + `test/` audio folders. If you unzipped the
> competition zip into `data/`, that is `data/SRCASW-BhavVaani` (or set
> `train_audio_dir` / `test_audio_dir` if you used a different layout).

## 1. Environment

```bash
# from the repo root
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt # or: pip install -r requirements.txt
```

Key packages: `torch>=2.6`, `transformers>=4.44`, `librosa`, `soundfile`,
`opensmile`, `scikit-learn`, `lightgbm`, `scipy`, `pyyaml`, `tqdm`, `matplotlib`.

## 2. Train an SSL model (5 folds)

```bash
python -m src.train_ssl --config configs/base_superb.yaml --folds 0 1 2 3 4 --out runs/superb
```

* `best.pt` per fold, per-epoch metrics in `runs/superb/fold{k}/train_log.csv`,
  OOF artefacts (`oof_probs.npy`, `oof_preds.csv`), CV summary in `runs/superb/cv_summary.json`.
* Any config key can be overridden on the CLI, so you never have to edit a YAML:

  ```bash
  python -m src.train_ssl --config configs/base_superb.yaml --folds 0 \
      --set data_root=data/SRCASW-BhavVaani max_seconds=4.0 batch_size=4
  ```

* Folds come from a shared manifest (`<data_root>/folds.json`, copied into the
  run directory). Override it explicitly with `--folds-json data/folds.json`
  (`train_ssl`, `predict` and `classic` all accept it).
* 6 GB GPU: configs ship batch 4 + `grad_accum 8` + gradient checkpointing for
  the *large* models, batch 8 + `grad_accum 4` for base models. On Kaggle
  (16 GB) use `batch_size: 16`, `grad_accum: 2` and `grad_ckpt: false`.
* Sanity check without data/GPU: `python -m src.train_ssl --config configs/base_superb.yaml --dry-run`.

## 3. Predict (OOF + test probabilities)

```bash
python -m src.predict --config configs/base_superb.yaml --ckpt runs/superb --out oof/superb
python -m src.predict --ckpt runs/superb --out oof/superb --tta      # 3-crop TTA
```

Writes `oof_probs_<tag>.csv` (filename, fold, true, prob_*) and
`test_probs_<tag>.csv` (filename, prob_angry, prob_happy, prob_neutral, prob_sad).

## 4. Classical baseline

```bash
python -m src.classic --config configs/classic.yaml          # eGeMAPS + MFCC -> SVM / LightGBM
```

Outputs `runs/classic/report.txt` plus ensemble-ready probability CSVs.
If `opensmile`/`lightgbm` is missing only that block is skipped — other modules
keep working (`pip install opensmile lightgbm`).

## 5. Ensemble + calibration

```bash
python -m src.ensemble --tags superb xlsr classic --out runs/final
```

Optimises non-negative weights on OOF macro-F1 (Nelder-Mead + greedy polish),
fits temperatures, then searches per-class prior/bias multipliers and a
smoothing floor. Writes `runs/final/test_probs.csv`, `weights.json`,
`score_report.txt`, `oof_blend.csv`.

Tags resolve to `oof/<tag>/oof_probs_<tag>.csv`; the classical baseline writes
its files into `runs/classic/`, so use `--oof-dir runs/classic --tags classic`
for it (or pass `--files <csv>...` explicitly).

## 6. Submission

```bash
python -m src.submit --probs runs/final/test_probs.csv          # -> submission.csv
python -m src.submit --probs runs/final/test_probs.csv --data-root /kaggle/input/<comp>
```

**The submitted file must use the exact header `Id,emotion` with lowercase
labels** (`angry`, `happy`, `neutral`, `sad`) — this is what
`sample_submission.csv` shows, and the Evaluation-page wording
(`AUDIO_ID,PREDICTED_LABEL`) is a red herring. The writer sorts rows by `Id`
and hard-validates: row count == `len(test.csv)`, `Id` set equality, label
vocabulary and lowercase-ness. It prints `RESULT: PASS/FAIL` and exits non-zero
on FAIL, so a broken file can never be uploaded silently. `--probs` accepts
either a probability CSV (`filename,prob_*`) or an already-built
`Id,emotion` file (handy for validating a hand-made baseline):

```bash
python -m src.submit --probs submission_baseline.csv --data-root data/SRCASW-BhavVaani
```

> Before submitting manually, always re-check the file header against
> `data/sample_submission.csv` (one `submission.csv` per Kaggle account, saved
> submissions are not scored, only the 2 selected ones are).

## 7. Kaggle notebooks

The notebook and the workstation share the same `src/`:

1. Add the competition data as a notebook input (accept the rules first).
2. Copy `src/` and `configs/` into the notebook (`git clone` the repo, or
   `cp` from a dataset you uploaded).
3. Detect the input path and generate a Kaggle config override:

   ```bash
   python -m src.kaggle_setup                              # -> configs/_kaggle.yaml
   python -m src.kaggle_setup --apply configs/xlsr.yaml   # -> configs/xlsr.kaggle.yaml
   ```

   `configs/xlsr.kaggle.yaml` is `extends: xlsr.yaml` + `data_root:
   /kaggle/input/<competition>` + `output_dir: /kaggle/working/runs`.
4. Train with the override, writing to `/kaggle/working`:

   ```bash
   python -m src.train_ssl --config configs/xlsr.kaggle.yaml --out /kaggle/working/runs/xlsr
   python -m src.predict --config configs/xlsr.kaggle.yaml --ckpt /kaggle/working/runs/xlsr \
       --out /kaggle/working/oof/xlsr
   ```

5. Download `oof/<tag>/*.csv` (or commit the Kaggle output as a dataset) so the
   ensemble can be built locally, or run `src.ensemble` + `src.submit` directly
   in the notebook.

Notes: keep `num_workers: 2` (notebooks dislike forked workers); the folds
manifest is written to `data/folds.json` locally and next to the run output on
Kaggle (read-only `/kaggle/input`); 30 h/week GPU budget goes a long way with
`max_seconds: 6`.

## 8. Module map

| Module | Purpose |
|---|---|
| `src/config.py` | `ExperimentConfig` dataclass + YAML (`extends:` supported) |
| `src/data.py` | CSV readers, audio loading/resampling/trim/pad |
| `src/splits.py` | stratified folds, persisted per filename |
| `src/augment.py` | speed/pitch/noise/SpecAugment + mixup |
| `src/dataset.py` | torch Dataset (waveform cache) + padding collate |
| `src/features.py` | openSMILE eGeMAPS + MFCC statistics (guarded imports) |
| `src/modeling.py` | wav2vec2/wavlm/hubert/whisper backbone + pooling head |
| `src/train_ssl.py` | fine-tuning CLI (AMP, grad-accum, early stop, OOF) |
| `src/predict.py` | OOF/test inference, optional TTA |
| `src/eval.py` | macro F1, per-class F1, confusion matrix, calibration search |
| `src/ensemble.py` | OOF weights + temperature + priors |
| `src/classic.py` | SVM / LightGBM baseline |
| `src/submit.py` | `submission.csv` writer + validator |
| `src/kaggle_setup.py` | Kaggle data-root detection / config override |

## 9. Troubleshooting

* **No audio found** — check `data_root` (the layout must contain
  `train.csv` + `train/` and `test.csv` + `test/`); `src.data.resolve_audio_path`
  also accepts flat or `audio/` layouts and logs missing files loudly.
* **OOM on 6 GB** — lower `batch_size`, raise `grad_accum`, keep `grad_ckpt: true`,
  or lower `max_seconds` (4 s still carries most emotion cues).
* **`opensmile` import error** — `pip install opensmile`; the SSL pipeline never
  needs it.
* **Reproducibility** — seeds are fixed (`seed: 42`); folds are reused from the
  manifest; full bitwise determinism is intentionally *not* forced (wav2vec2 uses
  non-deterministic kernels).
