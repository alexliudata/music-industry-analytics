# 📓 Jupyter Notebooks

This directory contains Jupyter notebooks for additional analysis and demonstrations.

## Available Notebooks

### `analysis_demo.ipynb`
A comprehensive demonstration notebook that shows:
- Data loading and preparation
- Genre trend analysis
- Artist performance evaluation
- Audio features correlation
- Business insights generation
- Custom visualizations
- Summary and next steps

## Running the Notebooks

1. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter
   ```

2. **Start Jupyter**:
   ```bash
   cd notebooks
   jupyter notebook
   ```

3. **Open the demo notebook**:
   - Navigate to `analysis_demo.ipynb`
   - Run cells sequentially to see the analysis in action

## Notebook Features

- **Interactive Analysis**: Run code cells to see results in real-time
- **Visualizations**: Plotly charts and matplotlib graphs
- **Data Exploration**: Sample data generation and analysis
- **Business Insights**: Strategic recommendations and risk assessment

## Example Usage

```python
# Load data
from src.data_collection import DataProcessor
processor = DataProcessor()
billboard_data = processor.load_data('sample_billboard_data.csv')

# Analyze trends
from src.analysis import GenreTrendAnalyzer
analyzer = GenreTrendAnalyzer(billboard_data, spotify_data)
emerging_genres = analyzer.identify_emerging_genres()
print(f"Emerging genres: {emerging_genres}")
```

## Tips

- Run cells in order for best results
- Modify parameters to explore different scenarios
- Export results for further analysis
- Use the insights for strategic decision-making 