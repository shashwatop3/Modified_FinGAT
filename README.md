# FinGAT: Financial Graph Attention Transformer

A deep learning framework for stock market prediction combining graph attention networks and transformer architectures.

Report File : https://www.overleaf.com/read/sqhjzrzcmsmc#d45cb9

## Overview

FinGAT (Financial Graph Attention Transformer) is a novel approach to predicting stock returns by leveraging both temporal patterns and industry sector relationships. The model processes historical stock data through a sophisticated pipeline of feature engineering, graph-based attention mechanisms, and transformer networks to generate actionable trading signals.

## Workflow

### 1. Data Preprocessing (`Preprocess.ipynb`)

- Load raw stock data and sector mappings
- Calculate technical indicators:
  - Simple Moving Averages (SMA)
  - Exponential Moving Averages (EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
- Generate advanced momentum indicators
- Create multi-index DataFrame with industry/stock structure
- Save processed data to parquet file

### 2. Model Training (`Transformer_FinGAT.ipynb`)

- Load preprocessed data
- Initialize the EnhancedFinGAT model with transformer architecture
- Train model on data up to 2023-12-31
- Validate on 2024 data
- Checkpoint best model based on validation performance
- Track metrics including Mean Reciprocal Rank (MRR), Mean Absolute Error (MAE), and movement accuracy

### 3. Model Testing (`test_set.ipynb`)

- Load trained model
- Evaluate on test period (Jan 11, 2025 - March 22, 2025)
- Calculate performance metrics:
  - Mean Reciprocal Rank (MRR)
  - Ranking precision
  - Investment Rate of Return (IRR)
  - Mean Absolute Error (MAE)
  - Movement accuracy
- Generate daily prediction CSVs for portfolio construction
- Analyze model effectiveness across different portfolio sizes (K=5, 10, 20)


### Output: Generated CSVs are stored in output_results.zip
### Ablation study in Ablation_study.pdf
