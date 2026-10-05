import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union
from .FactoST_finetune import FactoST
from ..callback.patch_mask import create_patch_history_st, create_patch_future_st
from .layers.utp import mask_rope_cos_sin
import math

class FactoST(FactoST):
    """FactoST: Learning to Factorize Spatio-Temporal Foundation Models.
    """
    
    def __init__(self, target_dim: int, seq_len: int, patch_len: int, stride: int, num_nodes: int, temporal_features: Dict[str, int], 
                 embedding_dim: int = 32,  use_st_filtering: Union[bool, Dict[str, bool]] = True, filter_matrices: List[str] = None, 
                 max_delay_steps: int = 3, use_st_metadata: bool = True, use_cpr: bool = True, num_token: int = 3, **kwargs):
        """Initialize FactoST model.
        """
        super().__init__(
            target_dim=target_dim, patch_len=patch_len,
            num_token=num_token, **kwargs
        )
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.stride = stride
        self.temporal_features = temporal_features
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.temporal_feature_names_map = {'minutes_of_hour': 0, 'time_of_day': 1, 'day_of_week': 2, 'day_of_month': 3, 'month_of_year': 4, 'spring_festival': 5, 'task_holiday': 6, 'statutory_date': 7, 'name': 8}
        self.temporal_feature_names = list(self.temporal_features.keys())
        self.use_st_metadata = use_st_metadata
        self.use_st_filtering = use_st_filtering if use_st_metadata else False
        self.use_cpr = use_cpr if use_st_metadata else False
        self.filter_matrices = filter_matrices or ['S_s', 'S_t', 'S_d']
        self.max_delay_steps = max_delay_steps
            
        # Initialize temporal queries for CPR
        if self.use_st_metadata and self.use_cpr:
            self.temporal_queries = nn.ParameterDict()
            self.temporal_aggregates = nn.ModuleDict()
                        
            for feature_name in self.temporal_features.keys():
                cycle_len = self.temporal_features[feature_name]
                self.temporal_queries[feature_name] = nn.Parameter(
                    torch.randn(cycle_len, self.d_model), requires_grad=True
                )
                self.temporal_aggregates[feature_name] = nn.MultiheadAttention(
                    embed_dim=self.d_model, num_heads=4, batch_first=True, dropout=0.5
                )
                self._init_weights(self.temporal_queries[feature_name])
                self._init_weights(self.temporal_aggregates[feature_name])
        
        # Initialize spatio-temporal metadata parameters
        if self.use_st_metadata:
            self.st_metadata_emb = nn.ModuleDict()
            
            # Node embeddings
            self.st_metadata_emb['node'] = nn.Embedding(num_embeddings=num_nodes, embedding_dim=self.embedding_dim)
            self._init_weights(self.st_metadata_emb['node'])
            
            # Temporal feature embeddings
            for feature_name, vocab_size in self.temporal_features.items():
                self.st_metadata_emb[feature_name] = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)
                self._init_weights(self.st_metadata_emb[feature_name])
            
            # Initialize projection layers for metadata fusion
            total_embedding_dim = self.embedding_dim * (1 + len(self.temporal_features))  # node + temporal features
            self.emb_proj = nn.Linear(total_embedding_dim, self.d_model)
            self.node_proj = nn.Linear(self.embedding_dim, self.d_model)
            self.temporal_proj = nn.Linear(self.embedding_dim, self.d_model)
            self._init_weights(self.emb_proj)
            self._init_weights(self.node_proj)
            self._init_weights(self.temporal_proj)
            
            # Parameters for delay mechanism
            self.gamma_lags = nn.Parameter(torch.ones(self.max_delay_steps))
            self.latent_prototypes = nn.Parameter(torch.randn(self.embedding_dim, self.d_model))
            self._init_weights(self.gamma_lags)
            self._init_weights(self.latent_prototypes)
            
            # Learnable weights for dynamic filter fusion
            self.score_weights = nn.ParameterDict({
                'S_s': nn.Parameter(torch.tensor(0.0)),
                'S_t': nn.Parameter(torch.tensor(0.0)),
                'S_d': nn.Parameter(torch.tensor(0.0))
            })
                
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Parameter):
            if module.dim() <= 1:
                nn.init.uniform(module, a=0.0, b=1.0)
            else:
                nn.init.xavier_uniform_(module)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def st_filtering(self, embeddings, batch_size, num_patches, st_metadata_embs, filter_matrices=None):
        """Apply spatio-temporal relation filtering using metadata.
        """
        # Defensive check for ablation studies
        if not filter_matrices:
            return embeddings

        n_vars = embeddings.size(0) // batch_size
        num_patch = num_patches
        d_model = embeddings.size(-1)
        
        try:
            embeddings_reshaped = embeddings.view(batch_size, n_vars, num_patch, d_model)

            # Get node embeddings (spatial)
            node_emb = st_metadata_embs['node'][:, :, :n_vars, :, :]  # [bs, num_patch, n_vars, patch_len, embedding_dim]
            node_emb_mean = node_emb.mean(dim=3)  # [bs, num_patch, n_vars, embedding_dim]
            node_emb_proj = self.node_proj(node_emb_mean)  # [bs, num_patch, n_vars, d_model]
            node_emb_proj = node_emb_proj.permute(0, 2, 1, 3)  # [bs, n_vars, num_patch, d_model]
            
            # Combine all temporal embeddings
            temporal_emb = torch.zeros_like(node_emb)
            for feature_name in self.temporal_features.keys():
                if feature_name in st_metadata_embs:
                    temporal_emb += st_metadata_embs[feature_name][:, :, :n_vars, :, :]
            temporal_emb_mean = temporal_emb.mean(dim=3)  # [bs, num_patch, n_vars, embedding_dim]
            temporal_emb_proj = self.temporal_proj(temporal_emb_mean)  # [bs, num_patch, n_vars, d_model]
            temporal_emb_proj = temporal_emb_proj.permute(0, 2, 1, 3)  # [bs, n_vars, num_patch, d_model]
            
            # Initialize attention scores
            node_scores = torch.zeros_like(embeddings_reshaped[..., 0])  # [bs, n_vars, num_patch]
            temporal_scores = torch.zeros_like(embeddings_reshaped[..., 0])  # [bs, n_vars, num_patch]
            delay_scores = torch.zeros_like(node_scores)  # [bs, n_vars, num_patch]
            
            # 1. Spatial attention (Identity Matching)
            if 'S_s' in filter_matrices:
                node_scores = torch.einsum('bnpd,bnpd->bnp', embeddings_reshaped, node_emb_proj)
            
            # 2. Temporal attention (Time Context Matching)
            if 'S_t' in filter_matrices:
                temporal_scores = torch.einsum('bnpd,bnpd->bnp', embeddings_reshaped, temporal_emb_proj)
            
            # 3. Delay attention (Causal Dependencies)
            if 'S_d' in filter_matrices:
                H = embeddings_reshaped  # [bs, n_vars, num_patch, d_model]
                max_k = min(self.max_delay_steps, H.size(2))
                gamma_lags = F.softplus(self.gamma_lags)  # [max_delay_steps]
                for k in range(1, max_k + 1):
                    H_lag = torch.zeros_like(H) 
                    H_lag[:, :, k:, :] = H[:, :, :-k, :] # Shift past to present position
                    proto_sim = torch.einsum('bnpd,md->bnpm', H_lag, self.latent_prototypes)
                    proto_weights = F.softmax(proto_sim, dim=-1)
                    H_agg_lag = torch.einsum('bnpm,md->bnpd', proto_weights, self.latent_prototypes)
                    delay_sim = torch.einsum('bnpd,bnpd->bnp', H, H_agg_lag)
                    gamma_k = gamma_lags[k - 1]
                    delay_scores += gamma_k * delay_sim
            
            # Collect active scores and their corresponding learnable weights
            scores = []
            active_weights = []
            for m in filter_matrices:
                if m == 'S_s':
                    scores.append(node_scores)
                    active_weights.append(self.score_weights['S_s'])
                elif m == 'S_t':
                    scores.append(temporal_scores)
                    active_weights.append(self.score_weights['S_t'])
                elif m == 'S_d':
                    scores.append(delay_scores)
                    active_weights.append(self.score_weights['S_d'])
                else:
                    raise ValueError(f"Unknown filter matrix name: {m}")
            
            weights = F.softmax(torch.stack(active_weights), dim=0) # [num_scores]
            fused_score = torch.stack(scores, dim=-1) @ weights
            combined_gate = torch.sigmoid(fused_score).unsqueeze(-1) # [bs, n_vars, num_patch, 1]
            return (embeddings_reshaped * combined_gate).reshape(batch_size * n_vars, num_patch, d_model)
            
        except Exception as e:
            print(f"Warning: Error in spatio-temporal filtering: {e}. Using original embeddings.")
            return embeddings

    def cyclic_prototype_refinement(self, z, history_data):
        """Cyclic Prototype Refinement (CPR): Refine temporal features using learnable cyclic prototypes.
        """
        bs, num_patch, n_vars, c, patch_len = history_data.shape
        
        # Flatten for processing: [Total_Tokens, 1, d_model]
        flat_z = z.reshape(-1, 1, self.d_model)
        
        # Apply refinement for each temporal cycle
        for i, feature_name in enumerate(self.temporal_feature_names):                
            indice = self.temporal_feature_names_map[feature_name]
            cycle_len = self.temporal_features[feature_name]
            time_idx = history_data[..., indice + 1, 0].long()  # [bs, num_patch, n_vars]
            time_idx = time_idx % cycle_len
            flat_time_idx = time_idx.reshape(-1)  # [Total_Tokens]
            query_prototype = self.temporal_queries[feature_name][flat_time_idx]  # [Total_Tokens, d_model]
            query_prototype = query_prototype.unsqueeze(1)  # [Total_Tokens, 1, d_model]
            attn_out, _ = self.temporal_aggregates[feature_name](query=query_prototype, key=flat_z, value=flat_z)
            flat_z = flat_z + attn_out
        
        # Restore shape: [bs, num_patch, n_vars, d_model]
        z_refined = flat_z.reshape(bs, num_patch, n_vars, self.d_model)
        return z_refined
    
    def st_metadata_fusion(self, history_data):
        """Spatio-Temporal Metadata Fusion (STMF) + Spatio-Temporal Filtering (STF).
        """
        bs, num_patch, n_vars, c, patch_len = history_data.shape
        
        # 1. Create metadata embeddings
        st_metadata_embs = {}
        
        # Spatial embeddings
        node_indices = torch.arange(self.num_nodes, device=self.st_metadata_emb['node'].weight.device)
        node_indices = node_indices.expand(bs, num_patch, patch_len, -1).transpose(-1, -2)
        st_metadata_embs['node'] = self.st_metadata_emb['node'](node_indices)
        
        # Temporal embeddings
        for i, feature_name in enumerate(self.temporal_feature_names):
            indice = self.temporal_feature_names_map[feature_name]
            cycle_len = self.temporal_features[feature_name]
            
            if i + 1 < c:
                feature_indices = history_data[..., indice + 1, :].long()
                feature_indices = feature_indices % cycle_len
                st_metadata_embs[feature_name] = self.st_metadata_emb[feature_name](feature_indices)
        
        # 2. Concatenate and project
        all_embs = [st_metadata_embs['node']]
        all_embs.extend([st_metadata_embs[feature_name] for feature_name in self.temporal_feature_names if feature_name in st_metadata_embs])
        combined_emb = torch.cat(all_embs, dim=-1)
        i_st = self.emb_proj(combined_emb)
        i_st = i_st.mean(dim=3)  # Pooling: [bs, num_patch, n_vars, d_model]
        
        # 3. Spatio-Temporal Filtering (STF): Filter I_st using affinities
        if self.use_st_filtering:
            i_st_reshaped = i_st.reshape(bs * n_vars, num_patch, self.d_model)
            filtered_emb = self.st_filtering(i_st_reshaped, bs, num_patch, st_metadata_embs, self.filter_matrices)
            i_st = filtered_emb.reshape(bs, num_patch, n_vars, self.d_model)
        
        return i_st

    def forward(self, x, future_temporal_features=None, **kwargs):
        """Forward pass with spatio-temporal features and backbone.
        """
        use_st_metadata = kwargs.get('use_st_metadata', self.use_st_metadata)
        use_cpr = kwargs.get('use_cpr', self.use_cpr)
        
        # Create patches for metadata fusion
        history_data, _ = create_patch_history_st(x, self.patch_len, self.stride)
        
        # Extract main input data
        x_main = x[..., 0] # [bs, seq_len, n_vars]
        bs, seq_len, n_vars = x_main.shape
        
        # Flatten for Channel Independence
        x_flat = x_main.permute(0, 2, 1).reshape(bs * n_vars, seq_len)
        
        mask = kwargs.get('mask', None)
        if mask is None:
            mask_flat = (~torch.isnan(x_flat)).float()
        else:
            mask_flat = mask.permute(0, 2, 1).reshape(bs * n_vars, seq_len)
            
        # 1. Prepare patched context
        patched_context, attention_mask, loc_scale = self.backbone._prepare_patched_context(x_flat, mask_flat)
        
        # 2. Prepare patched future
        num_output_patches = math.ceil(self.target_dim / self.patch_len)
        patched_future = self.backbone._prepare_patched_future(bs * n_vars, num_output_patches)
        
        # 3. Concatenate Context and Future
        patched_context_future = torch.cat([patched_context, patched_future], dim=1)
        
        # 4. Encode patches
        h = self.backbone.ts_encoder(patched_context_future) 
        
        # 5. Insert [REG] token
        num_input_patches = patched_context.shape[1]
        reg_token = self.backbone.reg_token.expand(bs * n_vars, -1, -1)
        h = torch.cat([h[:, :num_input_patches], reg_token, h[:, num_input_patches:]], dim=1)
        
        # 6. Spatio-temporal Adaptation
        if use_st_metadata:
            # Separate context and future
            h_context = h[:, :num_input_patches]
            h_future = h[:, num_input_patches+1:]
            
            # Prepare dimensions for Fusion: [bs, num_patches, n_vars, d_model]
            h_combined = torch.cat([h_context, h_future], dim=1)
            h_reshaped = h_combined.reshape(bs, n_vars, -1, self.d_model).permute(0, 2, 1, 3)
            
            # Future extrapolation for metadata
            future_patches = []
            
            if future_temporal_features is not None:
                # future_temporal_features: [bs, target_dim, features]
                # Expand to n_vars: [bs, target_dim, n_vars, features]
                if future_temporal_features.dim() == 3:
                     future_temporal_features = future_temporal_features.unsqueeze(2).expand(-1, -1, n_vars, -1)
                
                # Use create_patch_future_st to generate patches consistent with history logic
                # [bs, num_patches, n_vars, 1+features, patch_len]
                future_data = create_patch_future_st(future_temporal_features, self.patch_len, num_output_patches)

                # Ensure channel dimension matches history_data (handling cases where holidays are not used)
                c_hist = history_data.shape[3]
                if future_data.shape[3] > c_hist:
                    future_data = future_data[:, :, :, :c_hist, :]
                
                total_history_data = torch.cat([history_data, future_data], dim=1)
            else:
                # If future features are missing but required for metadata fusion, we cannot proceed.
                # User explicitly requested to raise an error (fallback to error) instead of extrapolating.
                raise ValueError("Future temporal features are required for FactoST when use_st_metadata=True.")


            # Cyclic Prototype Refinement (CPR): Refine Z to get Z'
            h_reshaped = self.cyclic_prototype_refinement(h_reshaped, total_history_data) if use_cpr else h_reshaped
            
            # Spatio-Temporal Metadata Fusion (STMF) + Filtering (STF)
            i_st = self.st_metadata_fusion(total_history_data)
            
            # Final combination: H_out = Z' + I_st
            h_reshaped = h_reshaped + i_st
            
            # Reshape back
            h_combined = h_reshaped.permute(0, 2, 1, 3).reshape(bs * n_vars, -1, self.d_model)
            h_context = h_combined[:, :num_input_patches]
            h_future = h_combined[:, num_input_patches:]
            
            h = torch.cat([h_context, h[:, num_input_patches:num_input_patches+1], h_future], dim=1)
        
        # 7. Domain Adaptation
        if self.num_token != 0:
            prompt = torch.mm(self.u, self.v)
            prompt = prompt.expand(bs * n_vars, -1, -1)
            adapted_prompt = self.domain_adapter(prompt)
            h = torch.cat((adapted_prompt, h), dim=1) 
            prompt_mask = torch.ones(bs * n_vars, self.num_token, dtype=torch.bool, device=h.device)
        else:
            prompt_mask = torch.tensor([], dtype=torch.bool, device=h.device).reshape(bs * n_vars, 0)
            
        # 8. Build Attention Mask [Prompt(1), Context(Mask), REG(1), Future(1)]
        attention_mask = torch.cat([
            prompt_mask,
            attention_mask,
            torch.ones(bs * n_vars, 1, dtype=torch.bool, device=attention_mask.device), # REG
            torch.ones(bs * n_vars, num_output_patches, dtype=torch.bool, device=attention_mask.device) # Future
        ], dim=1)
        
        # Expand mask for Transformer (B, 1, L, L)
        attention_mask_ext = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
        attention_mask_ext = attention_mask_ext.unsqueeze(1)
        
        # 9. Rotary Embedding & 10. Transformer Layers
        position_ids = torch.arange(0, h.shape[1], dtype=torch.long, device=h.device)
        position_ids = position_ids.unsqueeze(0).expand(bs * n_vars, -1)
        cos, sin = self.backbone.rotary_emb(h, position_ids)
        cos, sin = mask_rope_cos_sin(cos, sin, self.backbone.config.rope_percentage)
        
        for layer in self.backbone.encoder_layers:
            h = layer(h, (cos, sin), attention_mask_ext)
            
        # 11. Prediction Head
        outputs = self.backbone.predict_head(h[:, -num_output_patches:]) 
        outputs = outputs.reshape(bs * n_vars, num_output_patches * self.patch_len, -1)
        outputs = outputs[:, :self.target_dim, :]
        
        # 12. Denormalize
        loc, scale = loc_scale
        loc_scale_expanded = (loc.unsqueeze(1), scale.unsqueeze(1))
        outputs = self.backbone.instance_norm.inverse(outputs, loc_scale_expanded)
        
        # 13. Final Output
        try:
            median_idx = self.backbone.config.quantiles.index(0.5)
        except ValueError:
            median_idx = len(self.backbone.config.quantiles) // 2
            
        output = outputs[..., median_idx] 
        output = output.reshape(bs, n_vars, self.target_dim).permute(0, 2, 1)
        
        return output