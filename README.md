FinGAT Workflow
1. Data Preprocessing (Preprocess.ipynb)
Load raw stock data and sector mappings
Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
Generate advanced momentum indicators
Create multi-index DataFrame with industry/stock structure
Save processed data to parquet file
2. Model Training (Transformer_FinGAT.ipynb)
Load preprocessed data
Initialize the EnhancedFinGAT model with transformer architecture
Train model on data up to 2023-12-31
Validate on 2024 data
Checkpoint best model based on validation performance
3. Model Testing (test_set.ipynb)
Load trained model
Evaluate on test period (Jan 11, 2025 - March 22, 2025)
Calculate performance metrics (MRR, precision, IRR)
Generate daily prediction CSVs for portfolio construction
Analyze model effectiveness across different portfolio sizes
Each step builds on the previous one, creating a complete pipeline from raw data to actionable stock predictions.
