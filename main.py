import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import logging
from datetime import datetime
import json
from collections import defaultdict
import random
from math import sqrt
import copy

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.FileHandler('training_continual.log'),
       logging.StreamHandler()
   ]
)

class ConvLSTMCell(nn.Module):
   def __init__(self, in_channels, hidden_channels, kernel_size):
       super(ConvLSTMCell, self).__init__()
       self.hidden_channels = hidden_channels
       padding = kernel_size // 2
       self.gates = nn.Conv2d(
           in_channels + hidden_channels,
           4 * hidden_channels,
           kernel_size=kernel_size,
           padding=padding
       )

   def forward(self, x, hidden_state):
       hx, cx = hidden_state
       combined = torch.cat([x, hx], dim=1)
       gates = self.gates(combined)
       ingate, forgetgate, cellgate, outgate = torch.chunk(gates, 4, dim=1)
       ingate = torch.sigmoid(ingate)
       forgetgate = torch.sigmoid(forgetgate)
       cellgate = torch.tanh(cellgate)
       outgate = torch.sigmoid(outgate)
       cy = (forgetgate * cx) + (ingate * cellgate)
       hy = outgate * torch.tanh(cy)
       return hy, cy

class EnhancedConvLSTMPredictor(nn.Module):
   def __init__(self, input_shape=(3, 210, 160), sequence_length=50):
       super(EnhancedConvLSTMPredictor, self).__init__()
       self.sequence_length = sequence_length
       h, w = input_shape[1:]
       h = (h + 2 - 3) // 2 + 1
       w = (w + 2 - 3) // 2 + 1
       h = (h + 2 - 3) // 2 + 1
       w = (w + 2 - 3) // 2 + 1
       h = (h + 2 - 3) // 2 + 1
       w = (w + 2 - 3) // 2 + 1
       self.encoded_h = h
       self.encoded_w = w
       
       # Feature extraction backbone
       self.encoder = nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU()
       )
       
       # Temporal modeling with hierarchical ConvLSTM
       self.convlstm1 = ConvLSTMCell(64, 128, kernel_size=3)
       self.convlstm2 = ConvLSTMCell(128, 64, kernel_size=3)
       
       # Optical flow estimator (to detect motion patterns)
       self.flow_estimator = nn.Sequential(
           nn.Conv2d(64*2, 32, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Conv2d(32, 2, kernel_size=3, padding=1)  # 2 channels for x,y flow
       )
       
       # Frame reconstruction decoder
       self.decoder = nn.Sequential(
           nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, output_padding=(1,0)),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(1,0)),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, output_padding=(0,0)),
           nn.Sigmoid()
       )
       
       # Feature pooling for feature extraction
       self.feature_pooler = nn.Sequential(
           nn.AdaptiveAvgPool2d((4, 4)),
           nn.Flatten(),
           nn.Linear(64 * 4 * 4, 64)
       )
       
       # Remove specialized anomaly detection head for generic approach
       
       self.final_resize = nn.AdaptiveAvgPool2d((210, 160))
       self.init_hidden_states()
   
   def init_hidden_states(self):
       self.hidden1 = None
       self.hidden2 = None
   
   def reset_hidden_states(self, batch_size, device):
       self.hidden1 = (
           torch.zeros(batch_size, 128, self.encoded_h, self.encoded_w, device=device),
           torch.zeros(batch_size, 128, self.encoded_h, self.encoded_w, device=device)
       )
       self.hidden2 = (
           torch.zeros(batch_size, 64, self.encoded_h, self.encoded_w, device=device),
           torch.zeros(batch_size, 64, self.encoded_h, self.encoded_w, device=device)
       )

   def forward(self, x, reset_hidden=True, return_features=False):
       batch_size = x.size(0)
       device = x.device
       
       if reset_hidden:
           self.reset_hidden_states(batch_size, device)
       
       outputs = []
       features = []
       flow_outputs = []
       last_features = None
       
       for t in range(self.sequence_length):
           # Extract features
           current_features = self.encoder(x[:, t])
           
           # ConvLSTM processing
           h1, c1 = self.convlstm1(current_features, self.hidden1)
           h2, c2 = self.convlstm2(h1, self.hidden2)
           
           self.hidden1 = (h1, c1)
           self.hidden2 = (h2, c2)
           
           # Estimate optical flow if we have previous features
           if last_features is not None:
               flow = self.flow_estimator(torch.cat([h2, last_features], dim=1))
               flow_outputs.append(flow)
           else:
               flow_outputs.append(torch.zeros(batch_size, 2, h2.size(2), h2.size(3), device=device))
           
           # Store current features for next frame's flow estimation
           last_features = h2.detach()
           
           # Generate reconstructed frame
           decoded = self.decoder(h2)
           output = self.final_resize(decoded)
           outputs.append(output)
           
           # Extract pooled features
           pooled_feat = self.feature_pooler(h2)
           features.append(pooled_feat)
       
       # Stack outputs
       stacked_outputs = torch.stack(outputs, dim=1)
       stacked_features = torch.stack(features, dim=1)
       stacked_flows = torch.stack(flow_outputs, dim=1)
       
       if return_features:
           return stacked_outputs, stacked_features, stacked_flows
       else:
           return stacked_outputs

class CleanSequentialAtariDataset(Dataset):
    def __init__(self, folder_path, sequence_length=50, stride=25):
        self.folder_path = folder_path
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = []
        self.index_sequences()
        
    def index_sequences(self):
        episode_files = sorted([f for f in os.listdir(self.folder_path) 
                              if f.endswith('.hdf5') or f.endswith('.h5')])
        
        for episode_file in tqdm(episode_files, desc="Indexing clean sequences"):
            episode_path = os.path.join(self.folder_path, episode_file)
            try:
                with h5py.File(episode_path, 'r') as f:
                    n_frames = len(f['state'])
                    for i in range(0, n_frames - self.sequence_length - 1, self.stride):
                        self.sequences.append((episode_file, i))
            except Exception as e:
                logging.error(f"Error indexing episode {episode_file}: {str(e)}")
                continue
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        episode_file, start_idx = self.sequences[idx]
        episode_path = os.path.join(self.folder_path, episode_file)
        
        with h5py.File(episode_path, 'r') as f:
            frames = f['state'][start_idx:start_idx + self.sequence_length + 1]
            frames = torch.FloatTensor(frames).permute(0, 3, 1, 2) / 255.0
            
        input_seq = frames[:-1]
        target_seq = frames[1:]
        
        return input_seq, target_seq, episode_file

class EvalSequentialAtariDataset(Dataset):
    def __init__(self, folder_path, sequence_length=50, stride=25):
        self.folder_path = folder_path
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = []
        self.index_sequences()
        
    def index_sequences(self):
        episode_files = sorted([f for f in os.listdir(self.folder_path) 
                              if f.endswith('.hdf5') or f.endswith('.h5')])
        
        for episode_file in tqdm(episode_files, desc="Indexing sequences"):
            episode_path = os.path.join(self.folder_path, episode_file)
            try:
                with h5py.File(episode_path, 'r') as f:
                    n_frames = len(f['state'])
                    has_label = 'label' in f
                    has_tlabel = 'tlabel' in f
                    label_key = 'label' if has_label else ('tlabel' if has_tlabel else None)
                    
                    for i in range(0, n_frames - self.sequence_length - 1, self.stride):
                        self.sequences.append((episode_file, i, label_key))
            except Exception as e:
                logging.error(f"Error indexing episode {episode_file}: {str(e)}")
                continue
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        episode_file, start_idx, label_key = self.sequences[idx]
        episode_path = os.path.join(self.folder_path, episode_file)
        
        with h5py.File(episode_path, 'r') as f:
            frames = f['state'][start_idx:start_idx + self.sequence_length + 1]
            frames = torch.FloatTensor(frames).permute(0, 3, 1, 2) / 255.0
            
            if label_key and label_key in f:
                labels = f[label_key][start_idx:start_idx + self.sequence_length + 1]
                labels = torch.FloatTensor(labels)
            else:
                labels = torch.zeros(self.sequence_length + 1)
            
            input_seq = frames[:-1]
            target_seq = frames[1:]
            input_labels = labels[:-1]
            target_labels = labels[1:]
            
        return input_seq, target_seq, input_labels, target_labels, episode_file

def generic_evaluate_anomalies(model, eval_loader, device, meta_data, error_stats):
    """
    Generic evaluation function updated for model without anomaly detection head.
    Uses reconstruction error and temporal differences for anomaly detection.
    
    Args:
        model: The trained model
        eval_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        meta_data: Dictionary containing anomaly episode information
        error_stats: Dictionary with error statistics from clean data
        
    Returns:
        Dictionary with evaluation metrics for each anomaly type
    """
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    # Track scores and labels for each anomaly type
    anomaly_scores = defaultdict(list)
    anomaly_labels = defaultdict(list)
    episode_to_anomaly = {}
    
    # Create mapping of episodes to anomaly types
    for anomaly_type, episodes in meta_data['anomaly'].items():
        for episode in episodes:
            episode_to_anomaly[episode] = anomaly_type
    
    with torch.no_grad():
        for input_seq, target_seq, _, target_labels, episode_names in tqdm(eval_loader, desc="Evaluating"):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Get predictions - handle both model return types
            if hasattr(model, 'return_features') and model.return_features:
                predictions, features, flows = model(input_seq, return_features=True)
            else:
                predictions = model(input_seq)
            
            batch_size = input_seq.size(0)
            
            # Basic reconstruction error
            recon_errors = criterion(predictions, target_seq)
            recon_frame_errors = recon_errors.mean(dim=[2,3,4])
            
            # Temporal difference errors - general anomaly signal
            if predictions.size(1) > 1:  # Ensure we have multiple frames
                true_frame_diff = target_seq[:, 1:] - target_seq[:, :-1]
                true_frame_diff_mag = torch.abs(true_frame_diff).mean(dim=[2,3,4])
                
                pred_frame_diff = predictions[:, 1:] - predictions[:, :-1]
                pred_frame_diff_mag = torch.abs(pred_frame_diff).mean(dim=[2,3,4])
                
                # Calculate temporal difference error
                temp_diff_error = torch.abs(pred_frame_diff_mag - true_frame_diff_mag)
                # Pad to match sequence length
                temp_diff_error = F.pad(temp_diff_error, (0, 1), 'constant', 0)
            else:
                # If we only have one frame, use zeros for temporal error
                temp_diff_error = torch.zeros_like(recon_frame_errors)
            
            # Generic combined error signal based on reconstruction and temporal difference
            combined_errors = torch.zeros_like(recon_frame_errors)
            
            for t in range(combined_errors.size(1)):
                # Base reconstruction error
                base_error = recon_frame_errors[:, t]
                
                # Temporal error signal for all frames except first
                temp_error = temp_diff_error[:, t-1] if t > 0 else torch.zeros_like(base_error)
                
                # Combine errors with equal weighting
                combined_errors[:, t] = base_error + temp_error
            
            # Convert to numpy for easier processing
            frame_errors = combined_errors.cpu().numpy()
            
            # Process each sample in the batch
            for i, episode_name in enumerate(episode_names):
                anomaly_type = episode_to_anomaly.get(episode_name, 'normal')
                
                # Normalize errors using clean statistics
                mean_val = error_stats.get('mean', 0)
                std_val = error_stats.get('std', 1)
                normalized_errors = (frame_errors[i] - mean_val) / (std_val + 1e-10)
                
                # Store scores and labels
                anomaly_scores[anomaly_type].extend(normalized_errors.flatten())
                anomaly_labels[anomaly_type].extend(target_labels[i].numpy().flatten())
    
    # Calculate performance metrics
    results = {}
    for anomaly_type in anomaly_scores.keys():
        if anomaly_type != 'normal':
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(anomaly_labels[anomaly_type], anomaly_scores[anomaly_type])
            roc_auc = auc(fpr, tpr)
            
            # Calculate Precision-Recall curve and AUC
            precision, recall, _ = precision_recall_curve(
                anomaly_labels[anomaly_type],
                anomaly_scores[anomaly_type]
            )
            pr_auc = auc(recall, precision)
            
            # Store metrics
            results[anomaly_type] = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }
    
    return results


# Generic replay buffer without anomaly-specific sampling
class GenericReplayBuffer:
    def __init__(self, max_size_per_game=2000, importance_sampling=True):
        self.buffers = {}  # Samples by game
        self.max_size_per_game = max_size_per_game
        self.error_stats = defaultdict(lambda: {'mean': 0, 'std': 0, 'count': 0})
        self.importance_sampling = importance_sampling
        
    def add_sample(self, game_name, sample, error=None):
        # Add to buffer
        if game_name not in self.buffers:
            self.buffers[game_name] = []
        
        sample_data = {
            'input_seq': sample[0].cpu(),
            'target_seq': sample[1].cpu(),
            'error': error,
            'importance': 1.0  # Default importance
        }
        
        # Update error statistics
        if error is not None:
            stats = self.error_stats[game_name]
            old_mean = stats['mean']
            old_count = stats['count']
            new_count = old_count + 1
            
            # Update mean and std using Welford's online algorithm
            stats['count'] = new_count
            stats['mean'] = old_mean + (error - old_mean) / new_count
            if old_count > 0:
                stats['std'] = sqrt(
                    (stats['std']**2 * (old_count - 1) + 
                     (error - old_mean) * (error - stats['mean'])) / (new_count - 1)
                )
            
            # Generic importance based on error deviation from mean
            if stats['std'] > 0:
                z_score = abs((error - stats['mean']) / (stats['std'] + 1e-10))
                sample_data['importance'] = min(2.0, 0.5 + z_score)
        
        # Maintain buffer size with importance-based selection
        if len(self.buffers[game_name]) >= self.max_size_per_game:
            if self.importance_sampling:
                # Find the least important sample to replace
                min_idx = 0
                min_importance = float('inf')
                for i, s in enumerate(self.buffers[game_name]):
                    if s['importance'] < min_importance:
                        min_importance = s['importance']
                        min_idx = i
                        
                # Only replace if new sample is more important
                if sample_data['importance'] > min_importance:
                    self.buffers[game_name][min_idx] = sample_data
            else:
                # Standard reservoir sampling
                idx = random.randrange(len(self.buffers[game_name]))
                self.buffers[game_name][idx] = sample_data
        else:
            self.buffers[game_name].append(sample_data)
    
    def sample_from_previous_games(self, current_game, batch_size=8):
        samples = []
        
        # Collect previous games
        previous_games = [game for game in self.buffers.keys() if game != current_game]
        if not previous_games:
            return samples
        
        if self.importance_sampling:
            # Sample with importance weighting across all previous games
            all_samples = []
            all_weights = []
            
            for game in previous_games:
                game_samples = self.buffers[game]
                if game_samples:
                    all_samples.extend(game_samples)
                    all_weights.extend([s['importance'] for s in game_samples])
            
            if all_samples:
                # Normalize weights to probabilities
                probs = np.array(all_weights) / sum(all_weights)
                # Sample based on importance
                indices = np.random.choice(
                    len(all_samples), 
                    size=min(batch_size, len(all_samples)),
                    p=probs,
                    replace=False
                )
                samples = [all_samples[i] for i in indices]
        else:
            # Sample evenly from previous games
            samples_per_game = max(1, batch_size // len(previous_games))
            for game in previous_games:
                if self.buffers[game]:
                    game_samples = random.sample(
                        self.buffers[game],
                        min(samples_per_game, len(self.buffers[game]))
                    )
                    samples.extend(game_samples)
        
        return samples


# Generic EWC without specialized parameter weighting
class GenericEWC:
    def __init__(self, model, importance_scaling=500, importance_decay=0.7):
        """
        Generic EWC implementation without specialized parameter weighting.
        
        Args:
            model: The neural network model
            importance_scaling: Scaling factor for EWC penalty
            importance_decay: Decay factor for older tasks (0-1)
        """
        self.model = model
        self.importance_scaling = importance_scaling
        self.importance_decay = importance_decay
        self.fisher_dict = {}  # For each game: {parameter_name: fisher_values}
        self.optimal_params = {}  # For each game: {parameter_name: optimal_values}
        self.task_order = []  # Order of tasks/games for decay
        
    def update_fisher_params(self, game_name, train_loader, device):
        # Store current parameter values as optimal for this game
        self.optimal_params[game_name] = {}
        for n, p in self.model.named_parameters():
            self.optimal_params[game_name][n] = p.data.clone()
        
        # Initialize fisher values for this game
        self.fisher_dict[game_name] = {}
        for n, p in self.model.named_parameters():
            self.fisher_dict[game_name][n] = torch.zeros_like(p.data)
        
        # Add task to order list for decay
        if game_name not in self.task_order:
            self.task_order.append(game_name)
        
        # Compute fisher information matrix - generic implementation
        self.model.eval()
        
        for batch_idx, (input_seq, target_seq, _) in enumerate(tqdm(train_loader, desc="Computing Fisher")):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            self.model.zero_grad()
            predictions = self.model(input_seq)
            
            # Use reconstruction loss for Fisher computation
            loss = F.mse_loss(predictions, target_seq)
            loss.backward()
            
            # Accumulate squared gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher_dict[game_name][n] += p.grad.data ** 2 / len(train_loader)
        
        # Normalize fisher values to prevent extreme penalties
        self._normalize_fisher_values(game_name)
        
        # Apply decay to older tasks
        self._apply_task_decay()
    
    def _normalize_fisher_values(self, game_name):
        """Normalize Fisher values to prevent any parameters from dominating"""
        all_values = []
        for n in self.fisher_dict[game_name].keys():
            all_values.append(torch.max(self.fisher_dict[game_name][n]).item())
            
        if all_values:
            max_value = max(all_values)
            if max_value > 0:
                for n in self.fisher_dict[game_name].keys():
                    self.fisher_dict[game_name][n] /= (max_value + 1e-8)
    
    def _apply_task_decay(self):
        """Apply decay to older tasks' importance"""
        if self.importance_decay < 1.0:
            for i, task in enumerate(self.task_order[:-1]):  # All except the newest task
                # Calculate decay based on position (older tasks decay more)
                position_in_sequence = len(self.task_order) - i - 1
                decay_factor = self.importance_decay ** position_in_sequence
                
                # Apply decay to Fisher values
                for n in self.fisher_dict[task].keys():
                    self.fisher_dict[task][n] *= decay_factor
    
    def compute_consolidation_loss(self, current_game):
        ewc_loss = 0
        
        for game_name, fisher_values in self.fisher_dict.items():
            if game_name != current_game:  # Only consider previous games
                game_loss = 0
                for n, p in self.model.named_parameters():
                    if n in fisher_values and n in self.optimal_params[game_name]:
                        # Compute EWC penalty - all parameters weighted equally
                        param_loss = (fisher_values[n] * 
                                     (p - self.optimal_params[game_name][n]) ** 2).sum()
                        game_loss += param_loss
                
                ewc_loss += game_loss
        
        return self.importance_scaling * ewc_loss


# Generic training function without specialized loss components
def train_with_generic_continual_learning(model, train_loader, device, game_name, 
                                         replay_buffer, ewc, num_epochs=10):
    """Generic training with continual learning integration - updated for model without anomaly head"""
    # Configure learning rate
    is_first_game = len(ewc.fisher_dict) == 0
    initial_lr = 0.001 if is_first_game else 0.0005
    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Keep copy of model for knowledge distillation
    if len(replay_buffer.buffers) > 0:
        previous_model = copy.deepcopy(model)
        previous_model.eval()
    else:
        previous_model = None
    
    # Training hyperparameters
    accumulation_steps = 4
    num_previous_tasks = len(ewc.fisher_dict)
    replay_weight = min(0.5, 0.1 * num_previous_tasks)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        recon_losses = []
        temp_losses = []
        ewc_losses = []
        replay_losses = []
        batch_errors = []
        
        # Main training loop
        for batch_idx, (input_seq, target_seq, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Forward pass - updated for model without anomaly head
            if hasattr(model, 'return_features') and model.return_features:
                predictions, features, flows = model(input_seq, return_features=True)
            else:
                predictions = model(input_seq)
            
            # Reconstruction loss - main task objective
            recon_loss = criterion(predictions, target_seq)
            recon_losses.append(recon_loss.item())
            
            # Add temporal consistency loss
            temp_loss = 0
            if predictions.size(1) > 1:  # Ensure we have multiple frames
                true_frame_diff = target_seq[:, 1:] - target_seq[:, :-1]
                pred_frame_diff = predictions[:, 1:] - predictions[:, :-1]
                
                # Get magnitude of changes
                true_frame_diff_mag = torch.abs(true_frame_diff).mean(dim=[2,3,4])
                pred_frame_diff_mag = torch.abs(pred_frame_diff).mean(dim=[2,3,4])
                
                temp_loss = F.mse_loss(pred_frame_diff_mag, true_frame_diff_mag)
                temp_losses.append(temp_loss.item())
            
            # EWC loss for continual learning
            ewc_loss = ewc.compute_consolidation_loss(game_name)
            ewc_losses.append(ewc_loss.item() if ewc_loss > 0 else 0)
            
            # Combined task loss
            task_loss = recon_loss + temp_loss * 1.0 + ewc_loss
            
            # Add replay loss if we have previous tasks
            replay_loss = 0
            if previous_model and len(replay_buffer.buffers) > 0 and batch_idx % 2 == 0:
                replay_samples = replay_buffer.sample_from_previous_games(
                    game_name, batch_size=8
                )
                
                if replay_samples:
                    sample_losses = []
                    for sample in replay_samples:
                        replay_input = sample['input_seq'].to(device)
                        replay_target = sample['target_seq'].to(device)
                        sample_importance = sample['importance']
                        
                        # Get predictions from both models
                        with torch.no_grad():
                            if hasattr(previous_model, 'return_features') and previous_model.return_features:
                                prev_pred, prev_features, prev_flows = previous_model(replay_input, return_features=True)
                            else:
                                prev_pred = previous_model(replay_input)
                        
                        if hasattr(model, 'return_features') and model.return_features:
                            curr_pred, curr_features, curr_flows = model(replay_input, return_features=True)
                            
                            # Feature distillation if features are available
                            feature_distill_loss = F.mse_loss(curr_features, prev_features.detach())
                        else:
                            curr_pred = model(replay_input)
                            feature_distill_loss = 0
                        
                        # Reconstruction loss
                        replay_recon_loss = criterion(curr_pred, replay_target)
                        
                        # Prediction distillation loss
                        pred_distill_loss = criterion(curr_pred, prev_pred.detach())
                        
                        # Combined loss with importance weighting
                        sample_loss = (
                            replay_recon_loss + 
                            pred_distill_loss * 0.5 + 
                            feature_distill_loss * 0.5
                        ) * sample_importance
                        
                        sample_losses.append(sample_loss)
                    
                    if sample_losses:
                        replay_loss = sum(sample_losses) / len(sample_losses)
                    
                    replay_losses.append(replay_loss.item())
            
            # Total loss with replay
            loss = task_loss + (replay_weight * replay_loss)
            
            # Gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            # Save samples for replay buffer
            if batch_idx % 10 == 0:
                error_val = recon_loss.item()
                batch_errors.append(error_val)
                # Save sample
                replay_buffer.add_sample(
                    game_name,
                    (input_seq[0:1], target_seq[0:1]),
                    error_val
                )
        
        # Handle remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation loss
        val_loss = compute_validation_loss(model, train_loader, device, max_batches=50)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log progress
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1} - Game: {game_name}')
        logging.info(f'Average Loss: {avg_loss:.6f}, Validation Loss: {val_loss:.6f}')
        logging.info(f'Reconstruction Loss: {np.mean(recon_losses):.6f}')
        
        if temp_losses:
            logging.info(f'Temporal Loss: {np.mean(temp_losses):.6f}')
            
        logging.info(f'EWC Loss: {np.mean(ewc_losses):.6f}')
        
        if replay_losses:
            logging.info(f'Replay Loss: {np.mean(replay_losses):.6f}')
        
        if batch_errors:
            mean_error = np.mean(batch_errors)
            std_error = np.std(batch_errors)
            logging.info(f'Error Statistics - Mean: {mean_error:.6f}, Std: {std_error:.6f}')

    return val_loss

def compute_validation_loss(model, train_loader, device, max_batches=50):
    """Simple validation loss computation"""
    model.eval()
    val_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for input_seq, target_seq, _ in train_loader:
            if batch_count >= max_batches:
                break
                
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            predictions = model(input_seq)
            loss = F.mse_loss(predictions, target_seq)
            
            val_loss += loss.item()
            batch_count += 1
    
    return val_loss / max(1, batch_count)


def main_with_generic_continual_learning():
    """
    Main function for training and evaluating the model with generic continual learning
    strategies that are applicable to any scenario, not just specific anomaly types.
    """
    # List of games to learn sequentially
    games = ['BeamRiderNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'SeaquestNoFrameskip-v4']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with sequence length matching your data
    model = EnhancedConvLSTMPredictor(sequence_length=100).to(device)
    
    # Initialize generic replay buffer
    replay_buffer = GenericReplayBuffer(max_size_per_game=5000, importance_sampling=True)
    
    # Initialize generic EWC
    ewc = GenericEWC(
        model, 
        importance_scaling=700,
        importance_decay=0.7
    )
    
    # Create results directory
    results_dir = f'generic_cl700_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Track metrics across games
    all_metrics = {}
    
    # Loop through games
    for game_idx, game in enumerate(games):
        logging.info(f"\n{'='*50}")
        logging.info(f"Training on game {game_idx+1}/{len(games)}: {game}")
        logging.info(f"{'='*50}")
        
        # Create dataset and dataloader
        train_dataset = CleanSequentialAtariDataset(
            f'./viper_rl_data/datasets/atari/AAD/clean/{game}',
            sequence_length=100,
            stride=50
        )
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
        
        # Train with generic continual learning
        val_loss = train_with_generic_continual_learning(
            model, train_loader, device, game, replay_buffer, ewc, num_epochs=15
        )
        
        # Update fisher information
        ewc.update_fisher_params(game, train_loader, device)
        
        # Save model checkpoint
        torch.save({
            'game_idx': game_idx,
            'game_name': game,
            'model_state_dict': model.state_dict(),
            'ewc_state': {
                'fisher_dict': ewc.fisher_dict,
                'optimal_params': ewc.optimal_params,
                'task_order': ewc.task_order
            },
            'replay_buffer_stats': {game: replay_buffer.error_stats[game] for game in replay_buffer.error_stats}
        }, os.path.join(results_dir, f'model_after_{game}.pt'))
        
        # Evaluate on all games seen so far
        game_metrics = {}
        
        for eval_game_idx, eval_game in enumerate(games):
            logging.info(f"\nEvaluating on {eval_game} (Game {eval_game_idx+1}/{game_idx+1})")
            
            # Create evaluation dataset
            eval_dataset = EvalSequentialAtariDataset(
                f'./viper_rl_data/datasets/atari/AAD/anomaly/{eval_game}',
                sequence_length=100,
                stride=25
            )
            eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=8)
            
            # Load metadata
            eval_meta_path = f'./viper_rl_data/datasets/atari/AAD/anomaly/{eval_game}/meta.json'
            with open(eval_meta_path, 'r') as f:
                eval_meta_data = json.load(f)
            
            # Use generic evaluation
            results = generic_evaluate_anomalies(
                model,
                eval_loader,
                device,
                eval_meta_data,
                replay_buffer.error_stats.get(eval_game, {'mean': 0, 'std': 1})
            )
            
            # Save results
            results_file = os.path.join(
                results_dir,
                f'generic_cl_train_{game}_eval_{eval_game}.json'
            )
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Log results
            game_metrics[eval_game] = {}
            logging.info(f"\nResults for {eval_game}:")
            
            for anomaly_type, metrics in results.items():
                logging.info(f"{anomaly_type}:")
                logging.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
                logging.info(f"  PR AUC: {metrics['pr_auc']:.4f}")
                
                game_metrics[eval_game][anomaly_type] = metrics
        
        all_metrics[game] = game_metrics
    
    # Create summary
    summary_path = os.path.join(results_dir, 'summary_metrics.json')
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # Print forgetting analysis
    print("\n=== Catastrophic Forgetting Analysis ===")
    first_game = games[0]
    
    for anomaly_type in all_metrics[games[-1]][first_game]:
        print(f"\nTracking {anomaly_type} detection on {first_game}:")
        
        for game_idx, game in enumerate(games):
            if first_game in all_metrics[game]:
                roc_auc = all_metrics[game][first_game][anomaly_type]['roc_auc']
                print(f"  After training on {game} (task {game_idx+1}): ROC AUC = {roc_auc:.4f}")
    
    return all_metrics

if __name__ == "__main__":
    main_with_generic_continual_learning()