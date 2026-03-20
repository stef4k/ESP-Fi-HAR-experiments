import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import table4_benchmark as tb


def make_loeo_folds(manifest: pd.DataFrame, scenes_to_run: list[int] | None = None) -> list[dict]:
    if scenes_to_run is None:
        scenes_to_run = sorted(manifest["scene"].unique().tolist())
    scene_set = set(scenes_to_run)

    folds: list[dict] = []
    for target_scene in scenes_to_run:
        target_df = manifest[manifest["scene"] == target_scene]
        source_df = manifest[
            (manifest["scene"].isin(scene_set)) & (manifest["scene"] != target_scene)
        ]
        if target_df.empty:
            print(f"[warn] target scene={target_scene} missing in manifest, skipping.")
            continue
        if source_df.empty:
            print(f"[warn] no source-scene data for target scene={target_scene}, skipping.")
            continue

        folds.append(
            {
                "target_scene": target_scene,
                "target_scene_name": str(target_df["scene_name"].iloc[0]),
                "fold_id": f"scene_{target_scene}",
                "train": source_df,
                "test": target_df,
            }
        )

    return folds


def run_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenes_to_run = tb.parse_int_list(args.scenes)
    ml_models = tb.parse_str_list(args.ml_models)
    deep_models = tb.parse_str_list(args.deep_models)

    if args.run_ml and not ml_models:
        raise ValueError("ML run is enabled but no ML models were provided.")
    if args.run_deep and not deep_models:
        raise ValueError("Deep run is enabled but no deep models were provided.")

    unsupported_ml = sorted(set(ml_models) - set(tb.ML_HPARAMS))
    if unsupported_ml:
        raise ValueError(f"Unsupported ML models: {unsupported_ml}")
    if args.run_deep:
        unsupported_deep = sorted(set(deep_models) - set(tb.get_deep_model_builders()))
        if unsupported_deep:
            raise ValueError(f"Unsupported deep models: {unsupported_deep}")

    device = tb.resolve_compute_device(
        run_deep=args.run_deep, allow_cpu_fallback=args.allow_cpu_fallback
    )
    print("Device:", device)
    print("Dataset kind:", args.dataset_kind)
    print("Data root:", args.data_root)
    print("Scenes:", scenes_to_run)
    print("CV scheme: LOEO (leave-one-environment-out)")

    manifest, inferred_map, activity_order = tb.build_manifest(args.data_root, args.dataset_kind)
    print("Inferred activity-id mapping:", inferred_map)
    print("Activity order:", activity_order)
    tb.summarize_manifest(manifest)

    folds = make_loeo_folds(manifest, scenes_to_run=scenes_to_run)
    print("Total LOEO folds:", len(folds))
    if not folds:
        raise RuntimeError("No valid LOEO folds were generated. Check data_root and scenes.")

    results: list[dict] = []

    if args.run_ml:
        for model_name in ml_models:
            for fold in tqdm(folds, desc=f"ML {model_name}"):
                out = tb.train_eval_ml_fold(
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
                        "target_scene": fold["target_scene"],
                        "target_scene_name": fold["target_scene_name"],
                        "fold_id": fold["fold_id"],
                        **out,
                    }
                )

    if args.run_deep:
        for model_name in deep_models:
            for fold in tqdm(folds, desc=f"DL {model_name}"):
                out = tb.train_eval_deep_fold(
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
                        "target_scene": fold["target_scene"],
                        "target_scene_name": fold["target_scene_name"],
                        "fold_id": fold["fold_id"],
                        **out,
                    }
                )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No results were produced.")

    summary = (
        results_df.groupby(["model_type", "model", "target_scene", "target_scene_name"], as_index=False)
        .agg(
            folds=("fold_id", "nunique"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            f1_mean=("f1", "mean"),
        )
        .sort_values(["model_type", "model", "target_scene"])
        .reset_index(drop=True)
    )

    summary["acc_std"] = summary["acc_std"].fillna(0.0)
    for col in ["acc_mean", "acc_std", "f1_mean"]:
        summary[col] = summary[col] * 100.0

    return results_df, summary


def save_outputs(results_df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "loeo_fold_results.csv"
    summary_path = out_dir / "loeo_scene_summary.csv"
    summary_md_path = out_dir / "loeo_scene_summary.md"
    summary_txt_path = out_dir / "loeo_scene_summary.txt"

    results_df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    try:
        summary.round(2).to_markdown(summary_md_path, index=False)
        table_path = summary_md_path
    except ImportError:
        summary_txt_path.write_text(summary.round(2).to_string(index=False) + "\n", encoding="utf-8")
        table_path = summary_txt_path
        print(
            "[warn] Optional dependency 'tabulate' is not installed; "
            "saved plain-text table instead of markdown."
        )

    print("Saved:", raw_path)
    print("Saved:", summary_path)
    print("Saved:", table_path)


def build_arg_parser() -> argparse.ArgumentParser:
    this_dir = Path(__file__).resolve().parent
    default_data_root = Path(this_dir / "Data")
    default_out_dir = this_dir / "benchmark_outputs_loeo"

    parser = argparse.ArgumentParser(
        description="Run cross-environment LOEO benchmark on MAT or RF CSV datasets."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--output-dir", type=Path, default=default_out_dir)
    parser.add_argument(
        "--dataset-kind",
        type=str,
        choices=tb.SUPPORTED_DATASET_KINDS,
        default=tb.ESPFI_MAT_DATASET_KIND,
    )
    parser.add_argument("--scenes", type=str, default="1,2,3,4")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ml-n-jobs", type=int, default=1)
    parser.add_argument("--verbose-epochs", action="store_true")
    parser.add_argument("--ml-models", type=str, default=",".join(tb.DEFAULT_ML_MODELS))
    parser.add_argument("--deep-models", type=str, default=",".join(tb.DEFAULT_DEEP_MODELS))
    parser.add_argument("--run-ml", dest="run_ml", action="store_true")
    parser.add_argument("--no-ml", dest="run_ml", action="store_false")
    parser.set_defaults(run_ml=True)
    parser.add_argument("--run-deep", dest="run_deep", action="store_true")
    parser.add_argument("--no-deep", dest="run_deep", action="store_false")
    parser.set_defaults(run_deep=True)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    results_df, summary = run_benchmark(args)
    save_outputs(results_df, summary, args.output_dir)

    print("\nLOEO summary (rounded):")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
