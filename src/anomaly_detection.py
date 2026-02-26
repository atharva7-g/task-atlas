import random

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List
from src.parser import Parser
from src.utils import get_project_root


class AnomalyDetector:
	def __init__(self, parser: Parser, contamination=0.1, random_state=42, n_estimators=100):
		self.parser = parser
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
			n_estimators=self.n_estimators
		)

		return model

	def fit_predict(self, columns=None):
		if columns is None:
			columns = self.parser.get_metrics()

		X = self.parser.get_feature_matrix(columns)

		self.scaler = StandardScaler()
		X_scaled = self.scaler.fit_transform(X)

		self.model = IsolationForest(
			contamination=self.contamination,
			random_state=self.random_state,
			n_estimators=self.n_estimators,
		)

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

		result = self.parser.data[mask].copy()
		result["anomaly_score"] = self.scores[mask]

		result = result.sort_values("anomaly_score")

		if n is not None:
			result = result.head(n)

		return result


def main():
	parser = Parser(f"{get_project_root()}/archive/prmon.txt")
	parser.load()

	print(f"Loaded {len(parser.data)} rows")

	detector = AnomalyDetector(parser, contamination=0.05)
	detector.fit_predict()

	print(detector.get_anomaly_summary().to_string(index=False))

	top_anomalies = detector.get_anomalies(10)
	cols = ["Time", "wtime", "pss", "rss", "vmem", "rchar", "wchar", "anomaly_score"]
	print(top_anomalies[cols].to_string(index=False))

	anomalies = detector.get_anomalies()
	print(f"Found {len(anomalies)} anomalous rows")


if __name__ == "__main__":
	main()
