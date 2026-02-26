import os.path
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.parser import Parser
from src.utils import get_project_root


class AnomalyDetector:
	def __init__(
			self,
			filepath,
			contamination=0.1,
			random_state=42,
			n_estimators=100,
	):
		self.filepath = filepath
		self.parser = Parser(filepath)
		self.data = self.parser.data

		self.contamination = contamination
		self.random_state = random_state
		self.n_estimators = n_estimators

		self.scaler = None
		self.model = self._init_model()
		self.scores = None
		self.labels = None

	def _init_model(self):
		model = IsolationForest(
			contamination=self.contamination,
			random_state=self.random_state,
			n_estimators=self.n_estimators,
		)

		return model

	def fit_predict(self, columns=None):
		if columns is None:
			columns = self.parser.get_metrics()

		X = self.parser.get_feature_matrix(columns)
		self.scaler = StandardScaler()
		X_scaled = self.scaler.fit_transform(X)

		self.labels = self.model.fit_predict(X_scaled)
		self.scores = self.model.decision_function(X_scaled)

		return self.labels, self.scores

	def get_anomaly_summary(self) -> pd.DataFrame:
		anomaly_count = (self.labels == -1).sum()
		normal_count = (self.labels == 1).sum()

		summary = pd.DataFrame(
			{
				"total_rows": [len(self.labels)],
				"anomalies": [anomaly_count],
				"normal": [normal_count],
				"anomaly_rate": [anomaly_count / len(self.labels) * 100],
			}
		)
		return summary

	def get_anomalies(self, n=None):
		mask = self.labels == -1

		result = self.data[mask].copy()
		result["anomaly_score"] = self.scores[mask]

		result = result.sort_values("anomaly_score")

		if n is not None:
			result = result.head(n)

		return result

	def visualize(self, output_path="anomaly_visualization.png"):
		if "Time" in self.data.columns:
			self.data["Time"] = pd.to_datetime(self.data["Time"], unit="s")

		anomalous_cols = ['pss', 'rss', 'vmem', 'rchar', 'wchar', 'write_bytes', 'rx_packets', 'tx_bytes', 'tx_packets',
						  'stime', 'utime', 'nprocs', 'nthreads']

		n_cols = len(anomalous_cols)
		n_rows = (n_cols + 2) // 3

		fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
		axes = axes.flatten()

		x_data = self.data["Time"]
		anomaly_mask = self.labels == -1

		for idx, col in enumerate(anomalous_cols):
			ax = axes[idx]
			ax.plot(x_data, self.data[col], "b-", alpha=0.7, linewidth=1, label=col)

			anomaly_indices = np.where(anomaly_mask)[0]
			ax.scatter(
				x_data.iloc[anomaly_indices],
				self.data[col].iloc[anomaly_indices],
				c="red",
				s=20,
				zorder=5,
				label="anomaly",
			)

			ax.set_xlabel("Time")
			ax.set_ylabel(col)
			ax.set_title(col)
			ax.tick_params(axis="x", rotation=45)
			if idx == 0:
				ax.legend(loc="upper right", fontsize=8)

		for idx in range(n_cols, len(axes)):
			axes[idx].set_visible(False)

		plt.tight_layout()
		fig.savefig(output_path, dpi=150)
		print(f"Saved visualization to {output_path}")


def main():
	filepath = os.path.join(
		get_project_root(), "dataset", "prmon_combined_timeseries_sorted.tsv"
	)
	detector = AnomalyDetector(filepath)
	parser = Parser(filepath)

	print(f"Loaded {len(parser.data)} rows")
	print(f"Columns: {list(parser.data.columns)}")

	detector.fit_predict()

	print(detector.get_anomaly_summary().to_string(index=False))

	top_anomalies = detector.get_anomalies(10)
	cols = ["Time", "pss", "rss", "vmem", "rchar", "wchar", "anomaly_score"]
	print(top_anomalies[cols].to_string(index=False))

	anomalies = detector.get_anomalies()
	print(f"Found {len(anomalies)} anomalous rows")

	detector.visualize()

if __name__ == "__main__":
	main()
