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

# ===== HYPERPARAMETER TUNING =====

class FinGATHyperparameterTuner:
    """Hyperparameter tuning for the FinGAT model using Optuna"""

    def __init__(self, df, device, n_trials=30, study_name="fingat_hyperparameter_tuning", 
                 load_if_exists=True, sequence_length=15):
        self.df = df
        self.device = device
        self.n_trials = n_trials
        self.study_name = study_name
        self.load_if_exists = load_if_exists
        self.sequence_length = sequence_length
        self.best_params = None
        self.best_value = None
        self.study = None
        self.adv_momentum_index = None
        self.input_dim = None
        
    def prepare_data(self):
        """Prepare data and get necessary dimensions"""
        # Create small dataloaders for hyperparameter tuning
        train_loader, val_loader = create_dataloaders(
            df=self.df,
            batch_size=64,
            sequence_length=self.sequence_length
        )
        
        # Get input dimension and momentum feature index
        sample_batch = next(iter(train_loader))
        if sample_batch is not None:
            features, _, _, _, _, _ = sample_batch
            self.input_dim = features.shape[2]
            print(f"Input dimension: {self.input_dim}")
        else:
            self.input_dim = 32
            print(f"Using default input dimension: {self.input_dim}")
        
        # Get momentum feature index
        feature_cols = [col for col in self.df.columns.levels[2] if col != 'return_ratio']
        self.adv_momentum_index = feature_cols.index('adv_momentum')
        print(f"Advanced momentum feature found at index: {self.adv_momentum_index}")
        
        return train_loader, val_loader
    
    def ensure_on_device(self, module, device):
        """Ensure that a module and all its parameters are on the specified device"""
        if next(module.parameters(), None) is not None:
            if next(module.parameters()).device != device:
                module = module.to(device)
        return module


    def objective(self, trial):
        """Objective function for Optuna to optimize"""
        # Sample all parameters with fixed spaces
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128, 160, 256, 512])
        nhead = trial.suggest_categorical('nhead', [4, 8, 12])
        
        # Check if combination is valid immediately - early pruning for invalid combinations
        if hidden_dim % nhead != 0:
            # Return very poor score to discourage invalid combinations
            print(f"Trial {trial.number}: Invalid combination - hidden_dim={hidden_dim}, nhead={nhead}")
            return float('-inf')  # Worst possible score
        
        # Continue with the rest of the hyperparameters
        params = {
            'hidden_dim': hidden_dim,
            'nhead': nhead,
            'embed_dim': trial.suggest_categorical('embed_dim', [32, 48, 64, 12, 256]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.2),
            
            # Training parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
            'alpha': trial.suggest_float('alpha', 0.3, 0.7, step=0.1),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        }
        
        # Print current trial parameters
        print(f"\nTrial {trial.number}: Testing parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Create dataloaders with selected batch size
        train_loader, val_loader = create_dataloaders(
            df=self.df,
            batch_size=params['batch_size'],
            sequence_length=self.sequence_length
        )
        
        # Create model with trial hyperparameters
        model = EnhancedFinGAT(
            input_dim=self.input_dim,
            hidden_dim=params['hidden_dim'],
            embed_dim=params['embed_dim'],
            num_sectors=19,
            adv_momentum_index=self.adv_momentum_index
        ).to(self.device)
        
        # Create a new sequential learner with updated parameters and ensure it's on the correct device
        sequential_learner = EnhancedTransformerSequentialModule(
            input_dim=params['hidden_dim'],
            hidden_dim=params['hidden_dim'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(self.device)
        
        # Update the model's sequential learner
        model.sequential_learner = sequential_learner
        
        # Double-check that all model components are on the correct device
        for name, module in model.named_children():
            self.ensure_on_device(module, self.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Create loss function
        criterion = FinGATLoss(alpha=params['alpha'])
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Train for a few epochs
        epochs = 5  # Limited epochs for hyperparameter search
        trial_path = f"fingat_trial_{trial.number}.pt"
        
        # Add device verification before training
        sample_batch = next(iter(train_loader))
        if sample_batch is not None:
            features, sector_ids = sample_batch[0], sample_batch[1]
            features = features.to(self.device)
            sector_ids = sector_ids.to(self.device)
            
            # Verify device consistency before starting training
            print(f"Input device: {features.device}")
            for name, param in model.named_parameters():
                if param.device != features.device:
                    print(f"Warning: Parameter {name} is on {param.device}, but input is on {features.device}")
                    param.data = param.data.to(features.device)
        
        try:
            best_model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=epochs,
                device=self.device,
                scheduler=scheduler,
                patience=2,  # Small patience for quick trials
                save_path=trial_path
            )
            
            # Get validation metrics
            val_metrics = validate_model(best_model, val_loader, criterion, self.device)
            
            # Report metrics
            trial.set_user_attr('mrr@5', val_metrics.get('mrr@5', 0))
            trial.set_user_attr('mrr@15', val_metrics.get('mrr@15', 0))
            trial.set_user_attr('mrr@20', val_metrics.get('mrr@20', 0))
            trial.set_user_attr('movement_acc', val_metrics.get('movement_acc', 0))
            trial.set_user_attr('return_r2', val_metrics.get('return_r2', 0))
            
            # Return the MRR@20 as the optimization target
            return val_metrics.get('mrr@20', 0)
            
        except RuntimeError as e:
            # Handle device-related errors gracefully
            if "expected on mps" in str(e) or "expected on cuda" in str(e) or "device" in str(e).lower():
                print(f"Device error in trial {trial.number}: {e}")
                # Provide a fallback value so Optuna can continue with other trials
                return 0.0
            else:
                # Re-raise if it's not a device-related error
                raise e

    def run_tuning(self):
        """Run hyperparameter tuning"""
        print("Starting hyperparameter tuning...")
        
        # Prepare data
        self.prepare_data()
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(
            study_name=self.study_name,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=self.load_if_exists
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Store results
        self.study = study
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        # Print results
        print("\n" + "="*50)
        print("Hyperparameter tuning completed!")
        print(f"Best MRR@20: {self.best_value:.4f}")
        print("\nBest parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        # Save best hyperparameters
        self.save_best_params()
        
        return self.best_params, self.best_value
    
    def save_best_params(self, filepath=None):
        """Save best hyperparameters to a file"""
        if filepath is None:
            filepath = f"{self.study_name}_best_params.json"
            
        if self.best_params is None:
            print("No best parameters to save.")
            return
            
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=2)
            
        print(f"Best parameters saved to {filepath}")
        
        # Create and save visualization
        self.visualize_study(f"{self.study_name}_visualization.png")
        
    def load_best_params(self, filepath=None):
        """Load best hyperparameters from a file"""
        if filepath is None:
            filepath = f"{self.study_name}_best_params.json"
            
        try:
            with open(filepath, 'r') as f:
                self.best_params = json.load(f)
            print(f"Loaded best parameters from {filepath}")
            return self.best_params
        except FileNotFoundError:
            print(f"File {filepath} not found. Running hyperparameter tuning from scratch.")
            return None
    
    def visualize_study(self, filepath=None):
        """Visualize optimization results"""
        if self.study is None:
            print("No study to visualize.")
            return
            
        # Create plot
        fig = plt.figure(figsize=(12, 10))
        
        # Plot optimization history
        plt.subplot(2, 2, 1)
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        
        # Plot parameter importances
        plt.subplot(2, 2, 2)
        try:
            optuna.visualization.matplotlib.plot_param_importances(self.study)
        except:
            print("Could not plot parameter importances.")
            
        # Plot parallel coordinate
        plt.subplot(2, 2, 3)
        try:
            optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
        except:
            print("Could not plot parallel coordinate.")
            
        # Plot slice
        plt.subplot(2, 2, 4)
        try:
            optuna.visualization.matplotlib.plot_slice(self.study)
        except:
            print("Could not plot slice.")
            
        # Save figure
        if filepath is not None:
            plt.tight_layout()
            plt.savefig(filepath)
            print(f"Visualization saved to {filepath}")
            
        plt.close(fig)
        
    def create_model_with_best_params(self):
        """Create model with best hyperparameters"""
        if self.best_params is None:
            print("No best parameters available. Please run tuning first.")
            return None
            
        # Create model with best parameters
        model = EnhancedFinGAT(
            input_dim=self.input_dim,
            hidden_dim=self.best_params['hidden_dim'],
            embed_dim=self.best_params['embed_dim'],
            num_sectors=19,
            adv_momentum_index=self.adv_momentum_index
        ).to(self.device)
        
        # Update transformer parameters
        model.sequential_learner = EnhancedTransformerSequentialModule(
            input_dim=self.best_params['hidden_dim'],
            hidden_dim=self.best_params['hidden_dim'],
            nhead=self.best_params['nhead'],
            num_layers=self.best_params['num_layers'],
            dropout=self.best_params['dropout']
        )
        
        return model

# ===== MODEL COMPONENTS =====

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
    def __init__(self, input_dim, hidden_dim=64, embed_dim=64, num_sectors=19, adv_momentum_index=20):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_sectors = num_sectors
        self.adv_momentum_index = adv_momentum_index
        
        # Print configurations to debug dimension issues
        print(f"Model config: input_dim={input_dim}, hidden_dim={hidden_dim}, embed_dim={embed_dim}")
        
        # Feature extractors (unchanged)
        self.regular_feature_extractor = nn.Sequential(
            nn.Linear(input_dim-1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Momentum processing (unchanged - preserving look-ahead bias)
        self.momentum_extractor = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Direct prediction from momentum (unchanged)
        self.momentum_direct_predictor = nn.Linear(hidden_dim, embed_dim)
        
        # Feature fusion (unchanged)
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Temporal encoding (unchanged)
        self.temporal_encoder = SimpleTemporalEncoding(hidden_dim)
        
        # Enhanced sequential learning with improved transformer
        self.sequential_learner = EnhancedTransformerSequentialModule(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim,
            nhead=8,  # Increased number of heads
            num_layers=3  # Increased number of layers
        )
        
        # Momentum attention gate (unchanged)
        self.momentum_gate = nn.Sequential(
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
        
        # Task-specific prediction heads (unchanged)
        self.return_predictor = nn.Linear(embed_dim, 1)
        self.movement_predictor = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, sector_indices, historical_attentive=None, historical_graph=None):
        """Forward pass with unchanged interface to maintain compatibility"""
        batch_size, seq_len, _ = features.size()
        
        # 1. Extract the advanced indicator features and other features separately (unchanged)
        momentum_features = features[:, :, self.adv_momentum_index:self.adv_momentum_index+1]
        regular_features = torch.cat([
            features[:, :, :self.adv_momentum_index], 
            features[:, :, self.adv_momentum_index+1:]
        ], dim=2)
        
        # 2. Process features separately 
        regular_processed = self.regular_feature_extractor(regular_features)
        momentum_processed = self.momentum_extractor(momentum_features)
        momentum_signal = momentum_processed.mean(dim=1)  # Aggregate across time
        
        # Direct momentum influence (unchanged - preserving look-ahead bias)
        momentum_prediction = self.momentum_direct_predictor(momentum_signal)
        
        # 3. Concatenate features (unchanged)
        combined_features = torch.cat([regular_processed, momentum_processed], dim=2)
        features = self.feature_fusion(combined_features)
        
        # 4. Add temporal information (unchanged)
        features = self.temporal_encoder(features)
        
        # 5. Process sequence with enhanced transformer
        seq_context, seq_outputs = self.sequential_learner(features)
        
        # 6. Apply momentum attention gate (unchanged)
        gate_value = self.momentum_gate(seq_context)
        seq_context = seq_context * (2.0 + gate_value)  # Boost signal based on momentum
        
        # 7. Process intra-sector relationships with enhanced processing
        intra_sector_context = self.intra_sector_projection(seq_context)
        
        # 8. Inter-sector modeling with enhanced sector model
        sector_context = self.sector_model(intra_sector_context, sector_indices)
        
        # 9. Combine sequential and sector representations (unchanged)
        combined_context = torch.cat([seq_context, sector_context], dim=1)
        
        # 10. Fuse features (unchanged momentum influence - preserving look-ahead bias)
        fused_embeddings = self.fusion(combined_context)  
        fused_embeddings = fused_embeddings * 0.20 + momentum_prediction * 0.80
        
        # 11. Generate predictions (unchanged)
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

def collate_fn(batch):
    """Collate function for DataLoader"""
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Separate date elements
    features = torch.stack([item[0] for item in batch])
    sector_ids = torch.stack([item[1] for item in batch])
    company_ids = torch.stack([item[2] for item in batch])
    dates = [item[3] for item in batch]
    returns = torch.stack([item[4] for item in batch])
    movements = torch.stack([item[5] for item in batch])
    
    return features, sector_ids, company_ids, dates, returns, movements

def create_dataloaders(df, batch_size=64, sequence_length=15):
    """Creates train and validation DataLoaders with a chronological split"""
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Define split dates
    train_start = pd.Timestamp('2022-06-01')
    train_end = pd.Timestamp('2023-12-30')
    val_start = pd.Timestamp('2025-01-11')
    val_end = pd.Timestamp('2025-03-22')
    
    # Handle timezone if needed
    if df.index.tz is not None:
        try:
            train_start = train_start.tz_localize(df.index.tz)
            train_end = train_end.tz_localize(df.index.tz)
            val_start = val_start.tz_localize(df.index.tz)
            val_end = val_end.tz_localize(df.index.tz)
        except TypeError:
            train_start = train_start.tz_convert(df.index.tz)
            train_end = train_end.tz_convert(df.index.tz)
            val_start = val_start.tz_convert(df.index.tz)
            val_end = val_end.tz_convert(df.index.tz)
    
    print(f"Train period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
    print(f"Validation period: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
    
    # Split the data
    train_df = df.loc[train_start:train_end]
    val_df = df.loc[val_start:val_end]
    
    print(f"Training data size: {len(train_df)} rows")
    print(f"Validation data size: {len(val_df)} rows")
    
    # Create datasets
    train_dataset = FinGATDataset(
        multiindex_df=train_df,
        sequence_length=sequence_length
    )
    
    val_dataset = FinGATDataset(
        multiindex_df=val_df,
        sequence_length=sequence_length
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

# ===== TRAINING AND EVALUATION =====

def improved_mrr(return_preds, return_targets, company_ids=None, dates=None, top_ks=[5, 15, 20]):
    """
    Improved MRR calculation with proper date grouping
    
    Args:
        return_preds: Tensor of predicted returns
        return_targets: Tensor of actual returns
        company_ids: Optional tensor of company IDs (if None, will use indices)
        dates: Optional list of dates (if None, will calculate overall MRR)
        top_ks: List of k values for MRR@k calculation
    """
    mrr_metrics = {}
    
    # Case 1: If no dates provided, calculate global MRR
    if dates is None:
        # Sort predictions and targets to get ranking indices
        _, pred_indices = torch.sort(return_preds, descending=True)
        _, true_indices = torch.sort(return_targets, descending=True)
        
        # Create rank mapping
        pred_ranks = torch.zeros_like(pred_indices)
        for rank, idx in enumerate(pred_indices):
            pred_ranks[idx] = rank + 1  # 1-based ranking
        
        # Calculate MRR@k
        for k in top_ks:
            current_k = min(k, len(return_preds))

            if current_k == 0:
                mrr_metrics[f'mrr@{k}'] = 0.0
                continue
            
            mrr_sum = 0.0
            for i in range(current_k):
                if i < len(true_indices):
                    company_idx = true_indices[i]
                    rank = pred_ranks[company_idx].item()
                    
                    mrr_sum += 1.0 / rank
            
            mrr_metrics[f'mrr@{k}'] = mrr_sum / current_k

        return mrr_metrics
    
    # Case 2: With dates, calculate per-day MRR and average
    print("Using date-based MRR calculation")
    if isinstance(return_preds, torch.Tensor):
        return_preds = return_preds.cpu().numpy()
    if isinstance(return_targets, torch.Tensor):
        return_targets = return_targets.cpu().numpy()
    
    # Create company IDs if not provided
    if company_ids is None:
        company_ids = np.arange(len(return_preds))
        print(f"Created company_ids with shape: {company_ids.shape}")
    elif isinstance(company_ids, torch.Tensor):
        company_ids = company_ids.cpu().numpy()
        print(f"Converted company_ids to numpy with shape: {company_ids.shape}")
    
    # Create dataframe for date-based processing
    predictions_data = pd.DataFrame({
        'date': dates,
        'company_id': company_ids,
        'actual_return': return_targets,
        'predicted_return': return_preds
    })

    # Drop any rows with missing dates
    predictions_data.dropna(subset=['date'], inplace=True)

    # Group by date
    date_groups = predictions_data.groupby('date')
    print(f"Number of date groups: {len(date_groups)}")
    
    # Calculate MRR for each k value
    mrr_values = {k: [] for k in top_ks}
    
    for date, group in date_groups:
        batch_preds = torch.tensor(group['predicted_return'].values)
        batch_targets = torch.tensor(group['actual_return'].values)
        
        if len(batch_preds) == 0 or len(batch_targets) == 0:
            print(f"Skipping date {date} - empty data")
            continue
        
        # Sort by predictions and actual returns
        _, pred_indices = torch.sort(batch_preds, descending=True)
        _, true_indices = torch.sort(batch_targets, descending=True)
        
        # Calculate MRR for each k value
        for k in top_ks:
            current_k = min(k, len(batch_preds))

            if current_k == 0:
                mrr_values[k].append(0.0)
                print(f"  Skipping date {date} - current_k=0")
                continue
            
            mrr_day = 0.0
            
            for i in range(current_k):
                if i >= len(true_indices):
                    print(f"  Warning: i={i} exceeds true_indices length {len(true_indices)}")
                    continue
                    
                true_top_stock_idx = true_indices[i]
                pred_ranks = (pred_indices == true_top_stock_idx).nonzero(as_tuple=True)
                
                if len(pred_ranks) > 0 and len(pred_ranks[0]) > 0:
                    rank = pred_ranks[0][0].item() + 1  # 1-based ranking
                    mrr_day += 1.0 / rank
            
            daily_mrr = mrr_day / current_k
            mrr_values[k].append(daily_mrr)
    
    # Calculate average MRR across all days
    mrr_metrics = {f'mrr@{k}': np.mean(v) if v else 0.0 for k, v in mrr_values.items()}
    print(f"\nFinal MRR metrics: {mrr_metrics}")
    
    return mrr_metrics

def validate_model(model, dataloader, criterion, device, top_ks=[5, 15, 20]):
    """Validation function with progress bar"""
    model.eval()
    total_loss = 0.0
    all_return_preds = []
    all_return_targets = []
    all_movement_preds = []
    all_movement_targets = []
    all_company_ids = []
    all_dates = []
    batch_count = 0
    
    with torch.no_grad():
        # Use tqdm for validation progress
        val_progress = tqdm(
            dataloader, 
            desc="Validating",
            leave=False
        )
        
        for batch in val_progress:
            if batch is None:
                continue
            
            # Unpack batch
            features, sector_ids, company_ids, dates, returns, movements = batch
        
            # Move to device
            features = features.to(device)
            sector_ids = sector_ids.to(device)
            returns = returns.to(device)
            movements = movements.to(device)
            
            # Forward pass
            return_preds, movement_preds = model(features, sector_ids)
            
            # Calculate loss
            loss, _, _ = criterion(return_preds, returns, movement_preds, movements)
            
            # Track metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Store predictions
            all_return_preds.append(return_preds.cpu())
            all_return_targets.append(returns.cpu())
            all_movement_preds.append(movement_preds.cpu())
            all_movement_targets.append(movements.cpu())
            all_company_ids.append(company_ids.cpu())
            all_dates.extend(dates)
            
            # Update progress bar
            val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
    
    if batch_count == 0:
        return {'loss': float('inf')}
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, r2_score
    
    # Average loss
    avg_loss = total_loss / batch_count
    
    # Concatenate predictions
    all_return_preds = torch.cat(all_return_preds)
    all_return_targets = torch.cat(all_return_targets)
    all_movement_preds = torch.cat(all_movement_preds)
    all_movement_targets = torch.cat(all_movement_targets)
    all_company_ids = torch.cat(all_company_ids)
    
    # Calculate basic metrics
    return_r2 = r2_score(all_return_targets.numpy(), all_return_preds.numpy())
    movement_acc = accuracy_score(
        all_movement_targets.numpy() > 0.5, 
        all_movement_preds.numpy() > 0.5
    )
    
    # Calculate MRR
    mrr_metrics = improved_mrr(
        all_return_preds, 
        all_return_targets,
        company_ids=all_company_ids,
        dates=all_dates,
        top_ks=top_ks
    )
    
    # Combine metrics
    metrics = {
        'loss': avg_loss,
        'return_r2': return_r2,
        'movement_acc': movement_acc,
        **mrr_metrics
    }
    
    return metrics

def train_model(model, train_loader, val_loader, optimizer, criterion, 
                epochs, device, scheduler=None, patience=3, save_path='fingat_best.pt'):
    """Training function with progress bars"""
    print(f"Starting training on {device}")
    
    # Initialize tracking
    best_val_loss = float('inf')
    best_val_mrr = 0.0
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mrr@20': [], 'val_mrr@20': [],
        'train_acc': [], 'val_acc': []
    }
    
    # Main training loop with progress bar
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batch_count = 0
        
        # Use tqdm for batch progress
        train_progress = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            leave=False
        )
        
        for batch_idx, batch in train_progress:
            if batch is None:
                continue
            
            # Unpack batch
            features, sector_ids, company_ids, dates, returns, movements = batch
            
            # Move to device
            features = features.to(device)
            sector_ids = sector_ids.to(device)
            returns = returns.to(device)
            movements = movements.to(device)
            
            # Forward pass
            return_preds, movement_preds = model(features, sector_ids)
            
            # Calculate loss
            loss, _, _ = criterion(return_preds, returns, movement_preds, movements)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_batch_count += 1
            
            # Update progress bar
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average training loss
        avg_train_loss = train_loss / max(train_batch_count, 1)
        
        # Validation phase with progress bar
        print(f"Epoch {epoch+1}/{epochs} - Validating...")
        val_metrics = validate_model(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_mrr@20'].append(0.0)  # Placeholder
        history['val_mrr@20'].append(val_metrics.get('mrr@20', 0.0))
        history['train_acc'].append(0.0)  # Placeholder
        history['val_acc'].append(val_metrics.get('movement_acc', 0.0))
        
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Check if this is the best model
        current_mrr = val_metrics.get('mrr@20', 0.0)
        if current_mrr > best_val_mrr:
            best_val_mrr = current_mrr
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint at each improvement
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_mrr': best_val_mrr,
                'history': history,
                'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            }, save_path)
            
            improvement = "Improved! Model saved."
        else:
            patience_counter += 1
            improvement = f"No improvement, patience: {patience_counter}/{patience}"
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss: {val_metrics['loss']:.4f}, Val MRR@20: {val_metrics.get('mrr@20', 0.0):.4f}")
        print(f"  Val acc: {val_metrics.get('movement_acc', 0.0):.4f}")
        print(f"  {improvement}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with MRR@20: {best_val_mrr:.4f}")
    
    return model, history

def analyze_momentum_impact(model, val_loader, device):
    model.eval()
    momentum_correlations = []
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
                
            features, sector_ids, _, _, returns, _ = batch
            features = features.to(device)
            returns = returns.cpu().numpy()
            
            # Extract just momentum feature
            momentum_values = features[:, -1, model.adv_momentum_index].cpu().numpy()
            
            # Calculate correlation
            corr = np.corrcoef(momentum_values, returns)[0, 1]
            momentum_correlations.append(corr)
    
    print(f"Average momentum-return correlation: {np.mean(momentum_correlations):.4f}")

def save_model(model, history, filepath=None):
    """Save model with timestamp"""
    if filepath is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'fingat_model_{timestamp}.pt'
    
    # Save complete model information
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim,
            'embed_dim': model.embed_dim,
            'num_sectors': model.num_sectors,
            'adv_momentum_index': model.adv_momentum_index
        },
        'history': history,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }, filepath)
    
    print(f"Model saved to {filepath}")
    return filepath

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data_path = 'stock_data/processed/merged_stock_data_with_enhanced_features___.parquet'
    df = pd.read_parquet(data_path)
    
    # Add a parameter to control whether to perform hyperparameter tuning
    perform_tuning = True  # Set to False to skip tuning and use default or pre-tuned parameters
    
    if perform_tuning:
        # Create hyperparameter tuner
        tuner = FinGATHyperparameterTuner(
            df=df,
            device=device,
            n_trials=20,  # Adjust based on available compute resources
            study_name="fingat_hyperparameter_study",
            
        )
        
        # Try to load existing best parameters
        best_params = tuner.load_best_params()
        
        if best_params is None or input("Run hyperparameter tuning? (y/n): ").lower() == 'y':
            # Run hyperparameter tuning
            best_params, best_value = tuner.run_tuning()
        
        # Create model with best parameters
        model = tuner.create_model_with_best_params()
        
        # Create optimizer with tuned parameters
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )
        
        # Create loss function with tuned parameters
        criterion = FinGATLoss(alpha=best_params['alpha'])
        
    else:
        # Use default parameters (original code)
        # Get input dimension and momentum feature index
        train_loader, val_loader = create_dataloaders(
            df=df,
            batch_size=64,
            sequence_length=15
        )
        
        sample_batch = next(iter(train_loader))
        if sample_batch is not None:
            features, _, _, _, _, _ = sample_batch
            input_dim = features.shape[2]
            print(f"Input dimension: {input_dim}")
        else:
            input_dim = 32
            print(f"Using default input dimension: {input_dim}")
        
        # Get momentum feature index
        feature_cols = [col for col in df.columns.levels[2] if col != 'return_ratio']
        adv_momentum_index = feature_cols.index('adv_momentum')
        print(f"Advanced momentum feature found at index: {adv_momentum_index}")
        
        # Create model with default parameters
        model = EnhancedFinGAT(
            input_dim=input_dim,
            hidden_dim=128,
            embed_dim=64,
            num_sectors=19,
            adv_momentum_index=adv_momentum_index
        ).to(device)
        
        # Create optimizer with default parameters
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )
        
        # Create loss function with default parameters
        criterion = FinGATLoss(alpha=0.5)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    
    # Set model save path
    save_path = 'fingat_transformer_best.pt'
    
    # Train model
    print("\nStarting model training...")
    best_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=15,
        device=device,
        scheduler=scheduler,
        patience=3,
        save_path=save_path
    )
    
    # Save final model
    final_path = save_model(best_model, history, f'fingat_transformer_final.pt')
    
    # Analyze momentum impact
    analyze_momentum_impact(best_model, val_loader, device)
    
    print("Training complete!")