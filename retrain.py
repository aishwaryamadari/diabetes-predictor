"""
retrain.py — Standalone script to retrain the model pipeline.

Usage:
    python retrain.py
    python retrain.py --cv-folds 10
    python retrain.py --output-dir saved_model/v2
"""
import os
import sys
import json
import argparse
import logging
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main(cv_folds: int = 5, output_dir: str = "saved_model"):
    from ml.training import train_all
    from ml.dataset import load_pima_dataset

    logger.info("=" * 55)
    logger.info("  Diabetes Risk Predictor — Model Retraining")
    logger.info("=" * 55)

    df = load_pima_dataset()
    logger.info(f"Dataset: {len(df)} rows  |  {df.Outcome.sum()} diabetic ({df.Outcome.mean()*100:.1f}%)")

    os.makedirs(output_dir, exist_ok=True)

    pipeline, summary = train_all()

    logger.info("")
    logger.info("━" * 40)
    logger.info("  Model Comparison Summary")
    logger.info("━" * 40)
    fmt = "{:<22} {:>8} {:>8} {:>8} {:>8}"
    logger.info(fmt.format("Model", "Accuracy", "Precision", "F1", "AUC"))
    logger.info("─" * 60)
    for m in summary["models"]:
        marker = " ← best" if m["model"] == summary["best_model"] else ""
        logger.info(fmt.format(
            m["model"],
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['roc_auc']:.4f}",
        ) + marker)

    logger.info("")
    logger.info(f"✓  Best model : {summary['best_model']}  (AUC {summary['best_auc']:.4f})")
    logger.info(f"✓  Artifacts  : {output_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain the diabetes prediction model.")
    parser.add_argument("--cv-folds",   type=int, default=5,            help="Number of CV folds")
    parser.add_argument("--output-dir", type=str, default="saved_model", help="Output directory")
    args = parser.parse_args()
    main(cv_folds=args.cv_folds, output_dir=args.output_dir)
