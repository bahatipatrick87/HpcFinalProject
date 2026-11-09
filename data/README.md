# Data

The pipeline reads the EuroSAT dataset from `data/EuroSAT/` (set in `config/pipeline.yaml`):
27,000 RGB JPEGs, the same 27,000 patches as 13-band GeoTIFFs, and 597 further GeoTIFFs named
Jakarta and Bombay in `allBands/SeaLake` (`experiments/verify_dataset.py`). Download it from
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset and unzip it there.
With no images present, the pipeline falls back to simulated data.
