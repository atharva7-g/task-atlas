## Dataset

I was first told to generate a considerably large anomalous dataset. I used `prmon`'s `mem_burner` and `io_burner` to generate datasets in 6 execution phases: `baseline1`, `memory_spike`, `recovery1`, `cpu_surge`, `recovery2`, `io_burst`.

I used this template command in the python script located in `src/scripts` to generate this phasal dataset. 
```bash
./prmon --filename prmon_${phase_name}.txt --json-summary prmon_${phase_name}.json --interval 0.05 --
```


| Column Type | Columns |
|-------------|---------|
| **Non-numeric** | `phase` (6 values), `Time` (human-readable datetime) |
| **Numeric** | `wtime`, `gpufbmem`, `gpumempct`, `gpusmpct`, `ngpus`, `pss`, `rss`, `swap`, `vmem`, `rchar`, `read_bytes`, `wchar`, `write_bytes`, `rx_bytes`, `rx_packets`, `tx_bytes`, `tx_packets`, `stime`, `utime`, `nprocs`, `nthreads` |

### Running Anomaly Detection

Please install uv from: https://docs.astral.sh/uv/getting-started/installation/

```bash
uv run python -m src.anomaly_detection
```

### Output

Script loads dataset, runs `IsolationForest` with 10% contamination and prints anomalous records and statistics. Visualization is saved to `anomaly_visualization.png`.

## Code Snippets

### Detection Pipeline

```python
def fit_predict(self, columns=None):
    if columns is None:
        columns = self.parser.get_metrics()
    
    X = self.parser.get_feature_matrix(columns)
    self.scaler = StandardScaler()
    X_scaled = self.scaler.fit_transform(X)
    
    self.labels = self.model.fit_predict(X_scaled)
    self.scores = self.model.decision_function(X_scaled)
    
    return self.labels, self.scores
```

Uses sklearn's `StandardScaler` to normalize data. We then perform anomaly detection using sklearn's `IsolationForest`. 
### Visualization Method

```python
def visualize(self, output_path="anomaly_visualization.png"):
    self.data["Time"] = pd.to_datetime(self.data["Time"], unit="s")
    
    for idx, col in enumerate(anomalous_cols):
        ax.plot(x_data, self.data[col], "b-", alpha=0.7, label=col)
        anomaly_indices = np.where(self.labels == -1)[0]
        ax.scatter(x_data.iloc[anomaly_indices], 
                   self.data[col].iloc[anomaly_indices],
                   c="red", s=20, zorder=5, label="anomaly")
```

The visualization method creates a grid of subplots showing all monitored anomalous metrics. Please see output: `anomaly_visualization.png`

## Key Parameters

- contamination: expected proportion of anomalies in the dataset.
- random_state: seed
- n_estimators: number of Isolation Forest trees

Defaults generate around 408 anomalies. 

Please see `pyproject.toml` for dependencies. 

## Conclusions

Isolation Forest effectively detects anomalies in PrMon time series data, identifying unusual resource usage patterns across memory, CPU, I/O, and network metrics.
Isolation Forest is an unsupervised method and is able to characterize with significantly good accuracy, thanks in a good part to, the significantly easily characterizable dataset. 

# AI Usage

The scripts in `src/scripts` were written with the help of an AI assistant. I supplied the template function but asked an LLM to vary metrics to create anomalies. 
Additionally, the scatter plots in the visualization function were written with the help of AI as well. 

The subprocesses within the Jupyter Notebook used for visualization were written with the help of AI as well but this hasn't been used anywhere in the code. 

# References

1. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
2. https://www.kaggle.com/code/lkskrk/time-series-anomaly-detection-via-isolation-forest
