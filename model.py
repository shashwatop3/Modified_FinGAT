import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F
from torch import optim
from collections import defaultdict
import time
import datetime

# Device setup
torch._C._jit_set_bailout_depth(2)

if torch.backends.mps.is_available():
    print("MPS backend is available!")
    device = torch.device("mps")
else:
    print("MPS backend is not available. Using CUDA if available, otherwise CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
import matplotlib.pyplot as plt


class SimpleTemporalEncoding(nn.Module):
    """Simplified temporal encoding module"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # Create position embeddings to learn temporal patterns
        self.position_embeddings = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        # Add positional embeddings
        return x + self.position_embeddings[:seq_len]

class EnhancedTransformerSequentialModule(nn.Module):
    """Enhanced sequential learning module with dynamic transformer encoder"""
    def __init__(self, input_dim, hidden_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project input to hidden dimension if needed
        self.input_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Stack of transformer encoder blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attention': DynamicMultiHeadAttention(hidden_dim, nhead),
                'norm1': nn.LayerNorm(hidden_dim),
                'feedforward': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }))
        
        # Attention for sequence aggregation
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # Project input if dimensions don't match
        x = self.input_projection(x)
        
        # Process with transformer encoder blocks
        for layer in self.layers:
            # Self-attention with residual connection
            attn_output, _ = layer['attention'](x)
            x = layer['norm1'](x + attn_output)
            
            # Feed-forward with residual connection
            ff_output = layer['feedforward'](x)
            x = layer['norm2'](x + ff_output)
        
        # Apply attention for sequence aggregation
        attn_weights = F.softmax(self.attention(x), dim=1)
        context = torch.sum(x * attn_weights, dim=1)
        
        return context, x

class DynamicMultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with dynamic graph structure"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Make sure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        # Linear projections for queries, keys, values
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # Scaling factor for dot-product attention
        self.scaling = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        """
        x: Input of shape (batch_size, seq_len, embed_dim)
        mask: Optional mask to apply to attention scores
        """
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, values
        # Shape: (batch_size, seq_len, num_heads, head_dim)
        queries = self.query_projection(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key_projection(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_projection(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Calculate attention scores
        # scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        # context: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, values)
        
        # Transpose and reshape
        # output: (batch_size, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.output_projection(context)
        
        return output, attn_weights

class EnhancedSectorAttention(nn.Module):
    """Enhanced attention mechanism for sector modeling with graph awareness"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        # Multi-head attention components
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scaling = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, num_nodes, embed_dim]
        batch_size, num_nodes, _ = x.size()
        
        # Compute Q, K, V with multi-head separation
        q = self.query_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.key_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.value_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Reshape for attention computation
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_nodes, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Get attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, num_nodes, self.embed_dim)
        output = self.output_proj(context)
        
        return output

class EnhancedSectorModel(nn.Module):
    """Enhanced sector modeling component with graph-based attention"""
    def __init__(self, embed_dim, num_sectors, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sectors = num_sectors
        
        # Enhanced sector processor
        self.sector_attention = EnhancedSectorAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x_batch, sector_ids_batch):
        """Process a batch of embeddings with their sector IDs"""
        batch_size = x_batch.size(0)
        device = x_batch.device
        
        # Initialize sector representations
        sector_embeddings = torch.zeros(batch_size, self.num_sectors, self.embed_dim, device=device)
        sector_counts = torch.zeros(batch_size, self.num_sectors, 1, device=device)
        
        # Create sector mapping - each sample maps to one sector
        for i in range(batch_size):
            sector_id = sector_ids_batch[i].item()
            if 0 <= sector_id < self.num_sectors:
                sector_embeddings[i, sector_id] = x_batch[i]
                sector_counts[i, sector_id] = 1
        
        # Create attention mask for sectors that have at least one stock
        has_stocks = (sector_counts.squeeze(-1) > 0).float()
        mask = torch.bmm(has_stocks.unsqueeze(2), has_stocks.unsqueeze(1))
        
        # Process sectors - only apply attention where we have data
        attn_output = self.sector_attention(sector_embeddings, mask)
        refined_sector_embeddings = self.norm1(sector_embeddings + attn_output)
        
        # Apply feed-forward layer
        ff_output = self.feed_forward(refined_sector_embeddings)
        refined_sector_embeddings = self.norm2(refined_sector_embeddings + ff_output)
        
        # Extract the relevant sector embedding for each sample
        refined_x_batch = torch.zeros_like(x_batch)
        for i in range(batch_size):
            sector_id = sector_ids_batch[i].item()
            if 0 <= sector_id < self.num_sectors:
                refined_x_batch[i] = refined_sector_embeddings[i, sector_id]
            else:
                # Fall back to original embedding if sector is invalid
                refined_x_batch[i] = x_batch[i]
                
        return refined_x_batch

class EnhancedFinGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, embed_dim=64, num_sectors=19, technical_indicator=20):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_sectors = num_sectors
        self.technical_indicator = technical_indicator
        
        # Print configurations to debug dimension issues
        print(f"Model config: input_dim={input_dim}, hidden_dim={hidden_dim}, embed_dim={embed_dim}")
        
        # Feature extractors 
        self.regular_feature_extractor = nn.Sequential(
            nn.Linear(self.technical_indicator, hidden_dim),  
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        

        self.indicator_extractor = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Direct prediction from indicator 
        self.indicator_direct_predictor = nn.Linear(hidden_dim, embed_dim)
        
        # Feature fusion 
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Temporal encoding 
        self.temporal_encoder = SimpleTemporalEncoding(hidden_dim)
        
        # Enhanced sequential learning with improved transformer
        self.sequential_learner = EnhancedTransformerSequentialModule(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim,
            nhead=8,
            num_layers=3  # Increased number of layers
        )
        
        # indicator attention gate 
        self.indicator_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Enhanced intra-sector processing
        self.intra_sector_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # Using GELU instead of ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Enhanced inter-sector modeling
        self.sector_model = EnhancedSectorModel(hidden_dim, num_sectors, num_heads=4)
        
        # Enhanced fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)  # Adding dropout for regularization
        )
        
        # Task-specific prediction heads 
        self.return_predictor = nn.Linear(embed_dim, 1)
        self.movement_predictor = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, sector_indices, historical_attentive=None, historical_graph=None):
        """Forward pass with unchanged interface to maintain compatibility"""
        batch_size, seq_len, _ = features.size()
        
        indicator_features = features[:, :, self.technical_indicator:self.technical_indicator+1]
        regular_features = features[:, :, :self.technical_indicator]  # Removed trailing comma

        # 2. Process features separately 
        regular_processed = self.regular_feature_extractor(regular_features)
        indicator_processed = self.indicator_extractor(indicator_features)
        indicator_signal = indicator_processed.mean(dim=1)  # Aggregate across time
        
        indicator_prediction = self.indicator_direct_predictor(indicator_signal)
        
        # 3. Concatenate features 
        combined_features = torch.cat([regular_processed, indicator_processed], dim=2)
        features = self.feature_fusion(combined_features)
        
        # 4. Add temporal information 
        features = self.temporal_encoder(features)
        
        # 5. Process sequence with enhanced transformer
        seq_context, seq_outputs = self.sequential_learner(features)
        
        # 6. Apply indicator attention gate 
        gate_value = self.indicator_gate(seq_context)
        seq_context = seq_context * (2.0 + gate_value)  # Boost signal based on indicator
        
        # 7. Process intra-sector relationships with enhanced processing
        intra_sector_context = self.intra_sector_projection(seq_context)
        
        # 8. Inter-sector modeling with enhanced sector model
        sector_context = self.sector_model(intra_sector_context, sector_indices)
        
        # 9. Combine sequential and sector representations 
        combined_context = torch.cat([seq_context, sector_context], dim=1)
        

        fused_embeddings = self.fusion(combined_context)  
        fused_embeddings = fused_embeddings * 0.90 + indicator_prediction * 0.10
        
        # 11. Generate predictions 
        return_preds = self.return_predictor(fused_embeddings).squeeze(-1)
        movement_preds = self.movement_predictor(fused_embeddings).squeeze(-1)
        
        return return_preds, movement_preds
# ===== LOSS FUNCTION =====

class SimpleFocalLoss(nn.Module):
    """Simplified focal loss for classification"""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class FinGATLoss(nn.Module):
    """Loss function for FinGAT"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # Weight balance
        self.focal_loss = SimpleFocalLoss(gamma=2.0)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, return_preds, return_targets, movement_preds, movement_targets):
        # Calculate losses
        ranking_loss = self.mse_loss(return_preds, return_targets)
        movement_loss = self.focal_loss(movement_preds, movement_targets)
        
        # Combined loss with fixed weights
        combined_loss = self.alpha * ranking_loss + (1 - self.alpha) * movement_loss
        
        return combined_loss, ranking_loss, movement_loss

# ===== DATASET HANDLING =====

class FinGATDataset(Dataset):
    """Dataset for FinGAT"""
    def __init__(self, multiindex_df, sequence_length=5):
        self.sequence_length = sequence_length
        self.samples = []
        
        # Create mappings
        self.industry_map = {industry: idx for idx, industry in enumerate(multiindex_df.columns.levels[0])}
        self.company_map = {company: idx for idx, company in enumerate(multiindex_df.columns.levels[1])}
        
        # Ensure datetime index
        if not isinstance(multiindex_df.index, pd.DatetimeIndex):
            multiindex_df.index = pd.to_datetime(multiindex_df.index)
        
        # Store dates for reference
        self.dates = multiindex_df.index.tolist()
        
        # Process data by company
        for industry_id, industry in enumerate(multiindex_df.columns.levels[0]):
            for company_id, company in enumerate(multiindex_df.columns.levels[1]):
                try:
                    # Get company data
                    company_df = multiindex_df.xs((industry, company), axis=1, level=[0,1]).copy()
                    
                    if 'return_ratio' not in company_df.columns:
                        continue
                    
                    # Get features and labels
                    feature_cols = [col for col in company_df.columns if col != 'return_ratio']
                    if not feature_cols:
                        continue
                        
                    features = company_df[feature_cols].values
                    labels = company_df['return_ratio'].values
                    
                    # Create samples
                    for i in range(len(features) - self.sequence_length):
                        if features[i:i+self.sequence_length].size == 0:
                            continue
                            
                        self.samples.append({
                            'features': features[i:i+self.sequence_length],
                            'sector_id': self.industry_map[industry],
                            'company_id': self.company_map[company],
                            'date': multiindex_df.index[i + self.sequence_length],
                            'return_ratio': labels[i + self.sequence_length],
                            'movements': float(labels[i + self.sequence_length] > 0)
                        })
                
                except Exception as e:
                    continue
        
        # Sort by date
        self.samples.sort(key=lambda x: x['date'])
        
        # Check consistency
        if self.samples:
            self.feature_dim = self.samples[0]['features'].shape[1]
            self.samples = [s for s in self.samples if s['features'].shape[1] == self.feature_dim]
        else:
            self.feature_dim = 0
            print("Warning: No samples generated.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        features = torch.tensor(sample['features'], dtype=torch.float32)
        sector_id = torch.tensor(sample['sector_id'], dtype=torch.long)
        company_id = torch.tensor(sample['company_id'], dtype=torch.long)
        date = sample['date']
        return_ratio = torch.tensor(sample['return_ratio'], dtype=torch.float32)
        movements = torch.tensor(sample['movements'], dtype=torch.float32)
        
        return (
            features,
            sector_id,
            company_id,
            date,
            return_ratio,
            movements
        )
def custom_collate_fn(batch):
    """
    Custom collate function that handles timestamp objects
    
    Args:
        batch: List of tuples from the dataset
        
    Returns:
        Tuple of batched items, with timestamps kept as a list
    """
    # Separate different components
    features = [item[0] for item in batch]
    sector_ids = [item[1] for item in batch]
    company_ids = [item[2] for item in batch]
    dates = [item[3] for item in batch]  # Keep dates as a list
    return_ratios = [item[4] for item in batch]
    movements = [item[5] for item in batch]
    
    # Batch tensors normally
    features = torch.stack(features)
    sector_ids = torch.stack(sector_ids)
    company_ids = torch.stack(company_ids)
    return_ratios = torch.stack(return_ratios)
    movements = torch.stack(movements)
    
    # Return as a tuple
    return features, sector_ids, company_ids, dates, return_ratios, movements

def create_time_period_dataloaders(df, batch_size, sequence_length, start_date, end_date):
    """
    Create dataloaders for a specific time period
    
    Args:
        df: DataFrame with financial data
        batch_size: Batch size for dataloaders
        sequence_length: Sequence length for the model
        start_date: Start date for the period (inclusive)
        end_date: End date for the period (inclusive)
        
    Returns:
        DataLoader for the specified time period
    """
    # Convert to datetime if needed
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Handle timezone mismatch
    if df.index.tz is not None:
        # If DataFrame has timezone info, localize the input dates to match
        if start_date.tz is None:
            start_date = start_date.tz_localize(df.index.tz)
        if end_date.tz is None:
            end_date = end_date.tz_localize(df.index.tz)
    else:
        # If DataFrame is timezone naive, ensure input dates are also naive
        if start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if end_date.tz is not None:
            end_date = end_date.tz_localize(None)
    
    # Filter data by date range
    period_df = df.loc[start_date:end_date].copy()
    
    # Check if we have data in this period
    if period_df.empty:
        print(f"Warning: No data available between {start_date} and {end_date}")
        return None
    
    print(f"Creating dataloader for period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Number of trading days in period: {len(period_df)}")
    
    # Create dataset
    period_dataset = FinGATDataset(period_df, sequence_length=sequence_length)
    
    # Create dataloader with custom collate function
    period_loader = DataLoader(
        period_dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: don't shuffle time series data for evaluation
        num_workers=0,
        collate_fn=custom_collate_fn  # Add this line
    )
    
    return period_loader