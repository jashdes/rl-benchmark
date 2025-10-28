import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np


class ExperimentLogger:
    """Logs experiment configurations, metrics, and results."""

    def __init__(self, experiment_name, base_dir="experiments/results"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(base_dir) / f"{experiment_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = []
        self.config = {}

    def log_config(self, config):
        """Save experiment configuration."""
        self.config = config
        config_path = self.exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_path}")

    def log_metric(self, step, metric_name, value):
        """Log a single metric at a timestep."""
        self.metric.append({"step": step, "metric": metric_name, "value": float(value)})

    def log_final_results(self, results):
        """Save final evaluation results."""
        results_path = self.exp_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(self.metrics, f, indents=2)
        print(f"Results saved to {results_path}")

    def save_metrics(self):
        """Save all logged metrics."""
        if self.metrics:
            metrics_path = self.exp_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f)
            print(f"Metrics saved to {metrics_path}")

    def get_exp_dir(self):
        """Return experiment directory path."""
        return str(self.exp_dir)
