import argparse
import copy
import csv
import os
import random
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEFAULT_ACTIVITY_ORDER = ["arm_wave", "fall", "jump", "run", "squat", "turn", "walk"]
MAT_SCENE_NAME = {1: "Corridor", 2: "Office", 3: "Meeting room", 4: "Laboratory"}

DEFAULT_ACTIVITY_ID_TO_NAME = {
    1: "run",
    2: "walk",
    3: "jump",
    4: "squat",
    5: "arm_wave",
    6: "turn",
    7: "fall",
}

MAT_NAME_RE = re.compile(r"^(\d+)-(\d+)-(\d+)-(\d+)$")
RF_CSV_NAME_RE = re.compile(r"^E(\d+)_S(\d+)_A(\d+)_T(\d+)$")

ESPFI_MAT_DATASET_KIND = "espfi_mat"
RF_CSV_DATASET_KIND = "rf_csv"
SUPPORTED_DATASET_KINDS = [ESPFI_MAT_DATASET_KIND, RF_CSV_DATASET_KIND]

TARGET_TIME_STEPS = 950
TARGET_SUBCARRIERS = 52

DEFAULT_ML_MODELS = ["SVM", "RF", "LR"]
DEFAULT_DEEP_MODELS = [
    "LSTM",
    "GRU",
    "CNN",
    "ResNet18",
    "EfficientNetLite",
    "Transformer",
    "MobileNetV3",
]

# Aligned with notebook settings and ModelCode README guidance.
DEEP_HPARAMS = {
    "CNN": {"epochs": 50, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4},
    "ResNet18": {"epochs": 50, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4},
    "GRU": {"epochs": 100, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4},
    "LSTM": {"epochs": 100, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4},
    "Transformer": {"epochs": 100, "batch_size": 4, "lr": 1e-3, "weight_decay": 1e-4},
    "MobileNetV3": {"epochs": 50, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4},
    "EfficientNetLite": {"epochs": 50, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4},
}

ML_HPARAMS = {
    # Paper does not list exact classical-model hyperparameters.
    "SVM": {"C": 10.0, "kernel": "rbf", "gamma": "scale"},
    "RF": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    "LR": {"C": 1.0, "solver": "saga", "max_iter": 600},
}

PAPER_ROWS = [
    ("SVM", 1, 24.64, 8.01, 19.00),
    ("SVM", 2, 24.11, 7.09, 19.08),
    ("SVM", 3, 20.00, 9.20, 15.83),
    ("SVM", 4, 19.82, 10.61, 14.31),
    ("RF", 1, 23.57, 7.14, 19.01),
    ("RF", 2, 21.25, 6.16, 17.19),
    ("RF", 3, 24.64, 12.37, 21.82),
    ("RF", 4, 17.68, 7.59, 13.47),
    ("LR", 1, 19.29, 7.63, 16.87),
    ("LR", 2, 22.86, 3.98, 19.51),
    ("LR", 3, 16.43, 5.05, 12.66),
    ("LR", 4, 14.82, 2.36, 11.14),
    ("LSTM", 1, 32.15, 13.40, 29.82),
    ("LSTM", 2, 38.02, 14.21, 33.80),
    ("LSTM", 3, 36.20, 12.04, 33.84),
    ("LSTM", 4, 32.73, 13.25, 30.21),
    ("GRU", 1, 30.93, 3.78, 28.74),
    ("GRU", 2, 38.02, 8.63, 33.00),
    ("GRU", 3, 34.55, 4.21, 31.17),
    ("GRU", 4, 28.38, 8.98, 24.56),
    ("ResNet18", 1, 63.59, 14.91, 60.81),
    ("ResNet18", 2, 70.11, 19.60, 64.00),
    ("ResNet18", 3, 65.53, 18.22, 61.26),
    ("ResNet18", 4, 58.33, 22.73, 49.14),
    ("EfficientNetLite", 1, 61.60, 10.60, 57.13),
    ("EfficientNetLite", 2, 63.74, 8.74, 58.80),
    ("EfficientNetLite", 3, 61.98, 5.14, 57.93),
    ("EfficientNetLite", 4, 50.35, 6.96, 46.27),
    ("CNN", 1, 57.55, 11.43, 53.20),
    ("CNN", 2, 48.31, 8.02, 43.80),
    ("CNN", 3, 48.26, 11.14, 42.70),
    ("CNN", 4, 52.61, 12.19, 46.89),
    ("Transformer", 1, 54.43, 12.30, 49.85),
    ("Transformer", 2, 41.93, 9.47, 38.20),
    ("Transformer", 3, 47.51, 10.71, 42.43),
    ("Transformer", 4, 50.33, 15.11, 46.35),
    ("MobileNetV3", 1, 53.64, 13.24, 47.28),
    ("MobileNetV3", 2, 51.82, 11.85, 47.00),
    ("MobileNetV3", 3, 53.28, 15.18, 46.80),
    ("MobileNetV3", 4, 43.31, 16.25, 38.43),
]


@lru_cache(maxsize=1)
def get_deep_model_builders():
    from ESP_Fi_model import (
        CNN,
        ESP_Fi_GRU,
        ESP_Fi_LSTM,
        ESP_Fi_ResNet18,
        ESP_Fi_Transformer,
        EfficientNetLite,
        MobileNetV3,
    )

    return {
        "CNN": lambda num_classes: CNN(num_classes),
        "ResNet18": lambda num_classes: ESP_Fi_ResNet18(num_classes),
        "GRU": lambda num_classes: ESP_Fi_GRU(num_classes),
        "LSTM": lambda num_classes: ESP_Fi_LSTM(num_classes),
        "Transformer": lambda num_classes: ESP_Fi_Transformer(num_classes),
        "MobileNetV3": lambda num_classes: MobileNetV3(num_classes),
        "EfficientNetLite": lambda num_classes: EfficientNetLite(num_classes),
    }


def parse_int_list(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def normalize_activity_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")


def set_seed(seed: int = 666) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_compute_device(run_deep: bool, allow_cpu_fallback: bool = False) -> torch.device:
    if not run_deep:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("[warn] CUDA is not available. Deep models will run on CPU.")
        return torch.device("cpu")

    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
    device_name = torch.cuda.get_device_name(0)

    if arch_list and required_arch not in arch_list:
        message = (
            "Incompatible PyTorch CUDA build for the selected GPU.\n"
            f"GPU: {device_name} ({required_arch})\n"
            f"PyTorch: {torch.__version__}\n"
            f"torch.version.cuda: {torch.version.cuda}\n"
            f"Supported CUDA arch list in this build: {' '.join(arch_list)}\n"
            "Install a newer PyTorch build that supports sm_80 (A100), e.g.:\n"
            "python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu118 "
            "torch torchvision torchaudio"
        )
        if allow_cpu_fallback:
            print("[warn] " + message.replace("\n", " | "))
            print("[warn] Falling back to CPU because --allow-cpu-fallback is set.")
            return torch.device("cpu")
        raise RuntimeError(message)

    return torch.device("cuda")


def build_manifest_mat(data_root: Path) -> tuple[pd.DataFrame, dict[int, str], list[str]]:
    mat_files = sorted(data_root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under: {data_root}")

    inferred: dict[int, str] = {}
    rows = []

    for p in mat_files:
        m = MAT_NAME_RE.match(p.stem)
        if not m:
            continue

        scene, subject, activity_id, trial = map(int, m.groups())
        parent_name = normalize_activity_name(p.parent.name)

        if parent_name in DEFAULT_ACTIVITY_ORDER:
            if activity_id in inferred and inferred[activity_id] != parent_name:
                raise ValueError(
                    f"Conflicting mapping for activity_id={activity_id}: "
                    f"{inferred[activity_id]} vs {parent_name}"
                )
            inferred[activity_id] = parent_name

        rows.append(
            {
                "path": str(p),
                "scene": scene,
                "subject": subject,
                "activity_id": activity_id,
                "trial": trial,
                "parent_activity": parent_name,
            }
        )

    if not rows:
        raise RuntimeError("No valid X-Y-Z-M .mat files were parsed.")

    activity_id_to_name = dict(DEFAULT_ACTIVITY_ID_TO_NAME)
    activity_id_to_name.update(inferred)
    activity_order = list(DEFAULT_ACTIVITY_ORDER)

    clean_rows = []
    for r in rows:
        if r["parent_activity"] in activity_order:
            activity_name = r["parent_activity"]
        else:
            activity_name = activity_id_to_name.get(r["activity_id"])

        if activity_name not in activity_order:
            continue

        clean_rows.append(
            {
                **r,
                "activity_name": activity_name,
                "label": activity_order.index(activity_name),
                "scene_name": MAT_SCENE_NAME.get(r["scene"], f"Scene {r['scene']}"),
            }
        )

    manifest = pd.DataFrame(clean_rows).sort_values(
        ["scene", "subject", "activity_name", "trial", "path"]
    )
    manifest = manifest.reset_index(drop=True)

    if manifest.empty:
        raise RuntimeError("Manifest is empty after parsing labels.")

    return manifest, activity_id_to_name, activity_order


def build_manifest_rf_csv(data_root: Path) -> tuple[pd.DataFrame, dict[int, str], list[str]]:
    csv_files = sorted(data_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found under: {data_root}")

    inferred: dict[int, str] = {}
    rows = []

    for p in csv_files:
        m = RF_CSV_NAME_RE.match(p.stem)
        if not m:
            continue

        scene, subject, activity_id, trial = map(int, m.groups())
        parent_name = normalize_activity_name(p.parent.name)

        if activity_id in inferred and inferred[activity_id] != parent_name:
            raise ValueError(
                f"Conflicting mapping for activity_id={activity_id}: "
                f"{inferred[activity_id]} vs {parent_name}"
            )
        inferred[activity_id] = parent_name

        rows.append(
            {
                "path": str(p),
                "scene": scene,
                "subject": subject,
                "activity_id": activity_id,
                "trial": trial,
                "parent_activity": parent_name,
            }
        )

    if not rows:
        raise RuntimeError("No valid E#_S#_A##_T## .csv files were parsed.")

    if not inferred:
        raise RuntimeError("Could not infer activity mapping from RF CSV filenames.")

    activity_order = [inferred[k] for k in sorted(inferred.keys())]
    label_by_activity = {name: idx for idx, name in enumerate(activity_order)}

    clean_rows = []
    for r in rows:
        activity_name = r["parent_activity"]
        if activity_name not in label_by_activity:
            continue
        clean_rows.append(
            {
                **r,
                "activity_name": activity_name,
                "label": label_by_activity[activity_name],
                "scene_name": f"Environment {r['scene']}",
            }
        )

    manifest = pd.DataFrame(clean_rows).sort_values(
        ["scene", "subject", "activity_name", "trial", "path"]
    )
    manifest = manifest.reset_index(drop=True)

    if manifest.empty:
        raise RuntimeError("Manifest is empty after parsing labels.")

    return manifest, inferred, activity_order


def build_manifest(
    data_root: Path, dataset_kind: str
) -> tuple[pd.DataFrame, dict[int, str], list[str]]:
    if dataset_kind == ESPFI_MAT_DATASET_KIND:
        return build_manifest_mat(data_root)
    if dataset_kind == RF_CSV_DATASET_KIND:
        return build_manifest_rf_csv(data_root)
    raise ValueError(
        f"Unsupported dataset kind: {dataset_kind}. "
        f"Supported: {SUPPORTED_DATASET_KINDS}"
    )


def summarize_manifest(manifest: pd.DataFrame) -> None:
    print("Total samples:", len(manifest))
    print("Scenes:", sorted(manifest["scene"].unique().tolist()))
    print("Subjects:", sorted(manifest["subject"].unique().tolist()))
    print("Activities:", sorted(manifest["activity_name"].unique().tolist()))
    print("\nSamples per scene:")
    print(manifest.groupby(["scene", "scene_name"]).size().rename("n").reset_index())


def make_loso_folds(
    manifest: pd.DataFrame,
    scenes_to_run: list[int] | None = None,
    max_subject_folds: int | None = None,
) -> list[dict]:
    if scenes_to_run is None:
        scenes_to_run = sorted(manifest["scene"].unique().tolist())

    folds = []
    for scene in scenes_to_run:
        scene_df = manifest[manifest["scene"] == scene]
        if scene_df.empty:
            print(f"[warn] scene={scene} missing in manifest, skipping.")
            continue

        scene_name = str(scene_df["scene_name"].iloc[0])
        subjects = sorted(scene_df["subject"].unique().tolist())
        if max_subject_folds is not None:
            subjects = subjects[:max_subject_folds]

        for held_out_subject in subjects:
            train_df = scene_df[scene_df["subject"] != held_out_subject]
            test_df = scene_df[scene_df["subject"] == held_out_subject]
            if train_df.empty or test_df.empty:
                continue

            folds.append(
                {
                    "scene": scene,
                    "scene_name": scene_name,
                    "held_out_subject": held_out_subject,
                    "train": train_df,
                    "test": test_df,
                }
            )

    return folds


def resize_time_dimension(x: np.ndarray, target_steps: int = TARGET_TIME_STEPS) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array for resizing, got shape {x.shape}")
    num_steps, num_features = x.shape
    if num_steps == target_steps:
        return x
    if num_steps <= 0:
        raise ValueError("Cannot resize empty sequence.")
    if num_steps == 1:
        return np.repeat(x, target_steps, axis=0)

    old_axis = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    new_axis = np.linspace(0.0, 1.0, target_steps, dtype=np.float32)
    resized = np.empty((target_steps, num_features), dtype=np.float32)
    for i in range(num_features):
        resized[:, i] = np.interp(new_axis, old_axis, x[:, i])
    return resized


@lru_cache(maxsize=None)
def load_csi_sample_from_mat(path: str) -> np.ndarray:
    mat = sio.loadmat(path)
    key_candidates = ["CSIamp", "csi_amp", "CSI_amp", "amp"]
    data = None
    for key in key_candidates:
        if key in mat:
            data = mat[key]
            break

    if data is None:
        raise KeyError(f"No CSI amplitude key found in {path}. Tried: {key_candidates}")

    x = np.asarray(data, dtype=np.float32)
    if x.shape == (TARGET_TIME_STEPS, TARGET_SUBCARRIERS):
        pass
    elif x.shape == (TARGET_SUBCARRIERS, TARGET_TIME_STEPS):
        x = x.T
    else:
        raise ValueError(f"Unexpected sample shape {x.shape} in {path}")

    x = (x - x.mean()) / (x.std() + 1e-8)
    return x.reshape(1, TARGET_TIME_STEPS, TARGET_SUBCARRIERS).astype(np.float32)


def extract_amp_from_iq(serialized: str) -> np.ndarray:
    raw = serialized.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]

    iq = np.fromstring(raw, sep=",", dtype=np.float32)
    if iq.size == 0 or iq.size % 2 != 0:
        raise ValueError(
            "RF CSV row has malformed IQ payload; expected an even-length numeric list."
        )

    iq = iq.reshape(-1, 2)
    amp = np.sqrt(iq[:, 0] * iq[:, 0] + iq[:, 1] * iq[:, 1]).astype(np.float32)
    if amp.size > TARGET_SUBCARRIERS:
        amp = amp[:TARGET_SUBCARRIERS]
    elif amp.size < TARGET_SUBCARRIERS:
        amp = np.pad(amp, (0, TARGET_SUBCARRIERS - amp.size), mode="constant")
    return amp


@lru_cache(maxsize=None)
def load_csi_sample_from_rf_csv(path: str) -> np.ndarray:
    amp_rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "data" not in reader.fieldnames:
            raise KeyError(f"CSV does not include a 'data' column: {path}")

        for row in reader:
            if row.get("type") and row["type"] != "CSI_DATA":
                continue
            payload = row.get("data", "")
            if not payload:
                continue
            amp_rows.append(extract_amp_from_iq(payload))

    if not amp_rows:
        raise RuntimeError(f"No valid CSI rows found in: {path}")

    x = np.stack(amp_rows, axis=0).astype(np.float32)
    x = resize_time_dimension(x, target_steps=TARGET_TIME_STEPS)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x.reshape(1, TARGET_TIME_STEPS, TARGET_SUBCARRIERS).astype(np.float32)


@lru_cache(maxsize=None)
def load_csi_sample(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".mat":
        return load_csi_sample_from_mat(path)
    if suffix == ".csv":
        return load_csi_sample_from_rf_csv(path)
    raise ValueError(f"Unsupported sample file extension: {suffix} ({path})")


@lru_cache(maxsize=None)
def load_flat_feature(path: str) -> np.ndarray:
    return load_csi_sample(path).reshape(-1).astype(np.float32)


class ESPFiDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        x = load_csi_sample(row["path"]).copy()
        y = int(row["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def evaluate_deep(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return float(acc), float(f1)


def train_eval_deep_fold(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_classes: int,
    device: torch.device,
    seed: int = 666,
    num_workers: int = 2,
    verbose: bool = False,
) -> dict:
    set_seed(seed)
    hp = DEEP_HPARAMS[model_name]
    deep_model_builders = get_deep_model_builders()
    model = deep_model_builders[model_name](num_classes=num_classes).to(device)

    train_loader = DataLoader(
        ESPFiDataset(train_df),
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ESPFiDataset(test_df),
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp["epochs"]
    )

    best_train_acc = -1.0
    best_state = None

    for epoch in range(hp["epochs"]):
        model.train()
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        train_acc = correct / max(total, 1)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"[{model_name}] epoch {epoch + 1}/{hp['epochs']} train_acc={train_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    acc, f1 = evaluate_deep(model, test_loader, device)
    return {
        "acc": acc,
        "f1": f1,
        "best_train_acc": best_train_acc,
        "epochs": hp["epochs"],
        "batch_size": hp["batch_size"],
        "lr": hp["lr"],
        "weight_decay": hp["weight_decay"],
    }


def build_ml_model(model_name: str, seed: int = 666, n_jobs: int = 1):
    if model_name == "SVM":
        hp = ML_HPARAMS["SVM"]
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(C=hp["C"], kernel=hp["kernel"], gamma=hp["gamma"])),
            ]
        )

    if model_name == "RF":
        hp = ML_HPARAMS["RF"]
        return RandomForestClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"],
            random_state=seed,
            n_jobs=n_jobs,
        )

    if model_name == "LR":
        hp = ML_HPARAMS["LR"]
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=hp["C"],
                        solver=hp["solver"],
                        max_iter=hp["max_iter"],
                        random_state=seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported ML model: {model_name}")


def train_eval_ml_fold(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 666,
    n_jobs: int = 1,
) -> dict:
    set_seed(seed)

    x_train = np.stack([load_flat_feature(p) for p in train_df["path"].tolist()], axis=0)
    y_train = train_df["label"].to_numpy(dtype=np.int64)

    x_test = np.stack([load_flat_feature(p) for p in test_df["path"].tolist()], axis=0)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    model = build_ml_model(model_name, seed=seed, n_jobs=n_jobs)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    return {"acc": float(acc), "f1": float(f1)}


def build_paper_df() -> pd.DataFrame:
    paper_df = pd.DataFrame(
        PAPER_ROWS,
        columns=["model", "scene", "paper_acc_mean", "paper_acc_std", "paper_f1_mean"],
    )
    paper_df["scene_name"] = paper_df["scene"].map(MAT_SCENE_NAME)
    return paper_df


def empty_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_type",
            "model",
            "scene",
            "scene_name",
            "folds",
            "acc_mean",
            "acc_std",
            "f1_mean",
            "paper_acc_mean",
            "paper_acc_std",
            "paper_f1_mean",
            "delta_acc_mean",
            "delta_acc_std",
            "delta_f1_mean",
        ]
    )


def run_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenes_to_run = parse_int_list(args.scenes)
    ml_models = parse_str_list(args.ml_models)
    deep_models = parse_str_list(args.deep_models)

    if args.run_ml and not ml_models:
        raise ValueError("ML run is enabled but no ML models were provided.")
    if args.run_deep and not deep_models:
        raise ValueError("Deep run is enabled but no deep models were provided.")

    unsupported_ml = sorted(set(ml_models) - set(ML_HPARAMS))
    if unsupported_ml:
        raise ValueError(f"Unsupported ML models: {unsupported_ml}")
    if args.run_deep:
        unsupported_deep = sorted(set(deep_models) - set(get_deep_model_builders()))
        if unsupported_deep:
            raise ValueError(f"Unsupported deep models: {unsupported_deep}")

    device = resolve_compute_device(
        run_deep=args.run_deep, allow_cpu_fallback=args.allow_cpu_fallback
    )
    print("Device:", device)
    print("Dataset kind:", args.dataset_kind)
    print("Data root:", args.data_root)
    print("Scenes:", scenes_to_run)

    manifest, inferred_map, activity_order = build_manifest(args.data_root, args.dataset_kind)
    print("Inferred activity-id mapping:", inferred_map)
    print("Activity order:", activity_order)
    summarize_manifest(manifest)

    folds = make_loso_folds(
        manifest,
        scenes_to_run=scenes_to_run,
        max_subject_folds=args.max_subject_folds,
    )
    print("Total LOSO folds:", len(folds))
    if not folds:
        raise RuntimeError("No valid folds were generated. Check data_root and scenes.")

    results = []

    if args.run_ml:
        for model_name in ml_models:
            for fold in tqdm(folds, desc=f"ML {model_name}"):
                out = train_eval_ml_fold(
                    model_name=model_name,
                    train_df=fold["train"],
                    test_df=fold["test"],
                    seed=args.seed,
                    n_jobs=args.ml_n_jobs,
                )
                results.append(
                    {
                        "model_type": "ML",
                        "model": model_name,
                        "scene": fold["scene"],
                        "scene_name": fold["scene_name"],
                        "held_out_subject": fold["held_out_subject"],
                        **out,
                    }
                )

    if args.run_deep:
        for model_name in deep_models:
            for fold in tqdm(folds, desc=f"DL {model_name}"):
                out = train_eval_deep_fold(
                    model_name=model_name,
                    train_df=fold["train"],
                    test_df=fold["test"],
                    num_classes=len(activity_order),
                    device=device,
                    seed=args.seed,
                    num_workers=args.num_workers,
                    verbose=args.verbose_epochs,
                )
                results.append(
                    {
                        "model_type": "DL",
                        "model": model_name,
                        "scene": fold["scene"],
                        "scene_name": fold["scene_name"],
                        "held_out_subject": fold["held_out_subject"],
                        **out,
                    }
                )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No results were produced.")

    summary = (
        results_df.groupby(["model_type", "model", "scene", "scene_name"], as_index=False)
        .agg(
            folds=("held_out_subject", "nunique"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            f1_mean=("f1", "mean"),
        )
        .sort_values(["model_type", "model", "scene"])
        .reset_index(drop=True)
    )

    for col in ["acc_mean", "acc_std", "f1_mean"]:
        summary[col] = summary[col] * 100.0

    if args.compare_paper and args.dataset_kind == ESPFI_MAT_DATASET_KIND:
        paper_df = build_paper_df()
        comparison = summary.merge(
            paper_df, on=["model", "scene", "scene_name"], how="inner"
        )
        comparison["delta_acc_mean"] = comparison["acc_mean"] - comparison["paper_acc_mean"]
        comparison["delta_acc_std"] = comparison["acc_std"] - comparison["paper_acc_std"]
        comparison["delta_f1_mean"] = comparison["f1_mean"] - comparison["paper_f1_mean"]
        comparison = comparison.sort_values(["model_type", "model", "scene"]).reset_index(
            drop=True
        )
    else:
        comparison = empty_comparison_df()
        if args.compare_paper:
            print(
                "[info] Paper comparison skipped because it is only defined for "
                f"dataset kind '{ESPFI_MAT_DATASET_KIND}'."
            )

    return results_df, summary, comparison


def save_outputs(
    results_df: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "benchmark_fold_results.csv"
    summary_path = out_dir / "benchmark_scene_summary.csv"
    comparison_path = out_dir / "benchmark_vs_paper_table4.csv"
    comparison_md_path = out_dir / "benchmark_vs_paper_table4.md"
    comparison_txt_path = out_dir / "benchmark_vs_paper_table4.txt"

    results_df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    # pandas.to_markdown depends on optional package "tabulate".
    try:
        comparison.round(2).to_markdown(comparison_md_path, index=False)
        markdown_saved_path = comparison_md_path
    except ImportError:
        comparison_txt_path.write_text(
            comparison.round(2).to_string(index=False) + "\n",
            encoding="utf-8",
        )
        markdown_saved_path = comparison_txt_path
        print(
            "[warn] Optional dependency 'tabulate' is not installed; "
            "saved plain-text table instead of markdown."
        )

    print("Saved:", raw_path)
    print("Saved:", summary_path)
    print("Saved:", comparison_path)
    print("Saved:", markdown_saved_path)


def build_arg_parser() -> argparse.ArgumentParser:
    this_dir = Path(__file__).resolve().parent
    default_data_root = Path(os.environ.get("ESP_FI_DATA_ROOT", this_dir / "Data"))
    default_out_dir = this_dir / "benchmark_outputs"

    parser = argparse.ArgumentParser(
        description="Run ESP-Fi-style LOSO benchmark on MAT or RF CSV datasets."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--output-dir", type=Path, default=default_out_dir)
    parser.add_argument(
        "--dataset-kind",
        type=str,
        choices=SUPPORTED_DATASET_KINDS,
        default=ESPFI_MAT_DATASET_KIND,
    )
    parser.add_argument("--scenes", type=str, default="1,2,3,4")
    parser.add_argument("--max-subject-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ml-n-jobs", type=int, default=1)
    parser.add_argument("--verbose-epochs", action="store_true")
    parser.add_argument("--ml-models", type=str, default=",".join(DEFAULT_ML_MODELS))
    parser.add_argument("--deep-models", type=str, default=",".join(DEFAULT_DEEP_MODELS))
    parser.add_argument("--run-ml", dest="run_ml", action="store_true")
    parser.add_argument("--no-ml", dest="run_ml", action="store_false")
    parser.set_defaults(run_ml=True)
    parser.add_argument("--run-deep", dest="run_deep", action="store_true")
    parser.add_argument("--no-deep", dest="run_deep", action="store_false")
    parser.set_defaults(run_deep=True)
    parser.add_argument("--compare-paper", dest="compare_paper", action="store_true")
    parser.add_argument("--no-compare-paper", dest="compare_paper", action="store_false")
    parser.set_defaults(compare_paper=True)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    results_df, summary, comparison = run_benchmark(args)
    save_outputs(results_df, summary, comparison, args.output_dir)

    print("\nScene summary (rounded):")
    print(summary.round(2).to_string(index=False))
    if comparison.empty:
        print("\nComparison vs paper: skipped / empty for this dataset.")
    else:
        print("\nComparison vs paper (rounded):")
        print(comparison.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
