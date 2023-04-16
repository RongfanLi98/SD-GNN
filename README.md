This is a TensorFlow implementation of SD-GNN and TERME: GNN-Based Spatio-Temporal Manifold Learning: An Application of Landslide Prediction.

# GNN-Based Spatio-Temporal Manifold Learning: An Application of Landslide Prediction

More details of paper and dataset will be released after it is published.

# The Code
## Requirements

Following is the suggested way to install the dependencies:

    conda install --file environment.yaml

## Run the demo
```python
python main.py
```

All the parameter settings are in `utils.py`.

## Baselines

Our baselines included: 
1. History Average model (HA)
2. Autoregressive Integrated Moving Average model (ARIMA)
3. Support Vector Regression model (SVR)
4. Graph Convolutional Network model (GCN)
5. Gated Recurrent Unit model (GRU)
6. Slope-Aware Graph Neural Networks (SA-GNN)
7. STGCN (Wu et al.2020) and Point-GNN (Shi,Ragunathan, and Rajkumar 2020)

The python implementations of HA/ARIMA/SVR models are in the `baselines.py`. The GCN and GRU models are in `gcn.py` and `gru.py` respectively. Code of other baselines (STGCN, Point-GNN)  can be found in the corresponding papers.
