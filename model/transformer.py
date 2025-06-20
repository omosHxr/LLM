import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import json
import re
import hashlib
from pathlib import Path
import math
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Union, Dict, Any
import logging

# Enhanced configuration
CACHE_DIR = "search_cache"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='⚛️ [%(levelname)s] %(message)s')
logger = logging.getLogger("QuantumTransformer")

class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings with enhanced implementation"""
    def __init__(self, dim, max_len=512):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        
        # Create inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._build_cache(max_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len > self.max_len:
            self.max_len = seq_len
            self._build_cache(self.max_len)
        
        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )

def rotate_half(x):
    """Rotate half the channels"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class QuantumFourierFFN(nn.Module):
    """Quantum-inspired Fourier Feature Feed-Forward Network"""
    def __init__(self, dim, dropout=0.1, num_features=64):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        
        # Learnable Fourier features
        self.freq = nn.Parameter(torch.randn(1, num_features) * 10)
        self.phase = nn.Parameter(torch.randn(1, num_features))
        
        # Projection layers
        self.in_proj = nn.Linear(dim, num_features)
        self.out_proj = nn.Linear(num_features, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # Project to feature space
        x_proj = self.in_proj(x)
        
        # Apply Fourier transform
        fourier_features = torch.sin(x_proj * self.freq + self.phase)
        
        # Nonlinear transformation
        features = self.activation(fourier_features)
        
        # Project back to original dimension
        return self.dropout(self.out_proj(features))

class MultiHeadQuantumAttention(nn.Module):
    """Multi-head attention with rotary embeddings and KV caching"""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1 / math.sqrt(self.head_dim)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, query, key, value, mask=None, past_key_value=None, use_rotary_emb=False):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Handle past key values
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # Concatenate past keys/values with current
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # Save current key values for future use
        present_key_value = (k, v)
        key_len = k.size(2)
        
        # Apply rotary embeddings if requested
        if use_rotary_emb:
            cos, sin = self.rotary_emb(k, seq_dim=2)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Normalize
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Compute context
        context = torch.matmul(attn_probs, v)
        
        # Combine heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.dim)
        
        # Output projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return output, present_key_value

class QuantumTransformerLayer(nn.Module):
    """Enhanced transformer layer with quantum-inspired features"""
    def __init__(self, dim, heads, dropout=0.1, use_quantum_ffn=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.use_quantum_ffn = use_quantum_ffn
        
        # Attention components
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = MultiHeadQuantumAttention(dim, heads, dropout)
        
        # Feed-forward components
        self.ffn_norm = nn.LayerNorm(dim)
        if use_quantum_ffn:
            self.ffn = QuantumFourierFFN(dim, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, past_key_value=None, use_rotary_emb=False):
        # Attention block
        residual = x
        x = self.attn_norm(x)
        
        # Attention with possible past key values
        x, present_key_value = self.attn(
            x, x, x,
            mask=attention_mask,
            past_key_value=past_key_value,
            use_rotary_emb=use_rotary_emb
        )
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward block
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, present_key_value

class QuantumTransformer(nn.Module):
    def __init__(self, vocab_size=4096, dim=512, heads=8, layers=6, dropout=0.1, max_len=512, 
                 use_quantum_ffn=False, use_rotary_emb=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.max_len = max_len
        self.use_rotary_emb = use_rotary_emb
        self.dropout_rate = dropout
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Positional embeddings
        self.position = nn.Parameter(torch.zeros(1, max_len, dim))
        self.register_buffer('pos_enc', self._sinusoidal_encoding(max_len, dim))
        
        # Transformer encoder with quantum-inspired enhancements
        self.encoder_layers = nn.ModuleList([
            QuantumTransformerLayer(
                dim, heads, dropout, 
                use_quantum_ffn=use_quantum_ffn
            ) for _ in range(layers)
        ])
        
        # Normalization and output layers
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _sinusoidal_encoding(self, max_len, dim):
        """Create sinusoidal positional encoding"""
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pos_enc = torch.zeros(1, max_len, dim)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def _init_weights(self):
        """Advanced weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    nn.init.normal_(param, mean=0, std=0.02)
                elif 'norm' in name:
                    if param.ndim > 1:
                        nn.init.ones_(param)
                else:
                    if param.ndim > 1:
                        nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Special initialization for positional embeddings
        if hasattr(self, 'position'):
            nn.init.normal_(self.position, mean=0, std=0.02)

    def forward(self, x, attention_mask=None, past_key_values=None):
        # Embed tokens
        x = x.long()
        x = torch.clamp(x, 0, self.embedding.num_embeddings - 1)
        x = self.embedding(x)
        
        # Positional encoding
        seq_len = x.size(1)
        positions = self.position[:, :seq_len, :] + self.pos_enc[:, :seq_len, :]
        x = x + positions
        
        # Transformer processing with KV caching
        present_key_values = [] if past_key_values is None else past_key_values
        current_past = None
        all_present_key_values = []
        
        for i, layer in enumerate(self.encoder_layers):
            if past_key_values is not None:
                current_past = past_key_values[i] if i < len(past_key_values) else None
            
            x, kv_cache = layer(
                x, 
                attention_mask=attention_mask,
                past_key_value=current_past,
                use_rotary_emb=self.use_rotary_emb
            )
            all_present_key_values.append(kv_cache)
        
        # Output projection
        x = self.norm(x)
        logits = self.fc(x)
        return logits, all_present_key_values

    def _create_attention_mask(self, x, past_length=0):
        """Create causal attention mask with past key values support"""
        seq_len = x.size(1)
        total_len = past_length + seq_len
        
        # Create a lower triangular matrix for causal attention
        mask = torch.tril(torch.ones((seq_len, total_len), device=self.device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        return mask

    def expand_vocab(self, new_vocab_size, tokenizer=None):
        """Dynamically expand vocabulary embeddings with enhanced initialization"""
        device = self.device
        
        # Save original weights
        old_emb = self.embedding.weight.data
        old_fc_weight = self.fc.weight.data
        old_fc_bias = self.fc.bias.data
        
        # Create new embedding layer with smart initialization
        self.embedding = nn.Embedding(new_vocab_size, self.dim).to(device)
        
        # Initialize new embeddings with average of existing embeddings
        avg_embedding = torch.mean(old_emb, dim=0, keepdim=True)
        self.embedding.weight.data[:len(old_emb)] = old_emb
        self.embedding.weight.data[len(old_emb):] = avg_embedding
        
        # Create new output layer
        self.fc = nn.Linear(self.dim, new_vocab_size).to(device)
        self.fc.weight.data[:len(old_fc_weight)] = old_fc_weight
        
        # Initialize new weights with small random values
        new_weight = torch.empty(new_vocab_size - len(old_fc_weight), self.dim, device=device)
        nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
        self.fc.weight.data[len(old_fc_weight):] = new_weight
        
        # Initialize new biases with zeros
        self.fc.bias.data[:len(old_fc_bias)] = old_fc_bias
        self.fc.bias.data[len(old_fc_bias):] = 0
        
        # Update internal state
        self.vocab_size = new_vocab_size
        logger.info(f"Expanded vocabulary to {new_vocab_size} tokens")
        
        # Update tokenizer if provided
        if tokenizer and hasattr(tokenizer, 'vocab_size'):
            tokenizer.vocab_size = new_vocab_size
            logger.info(f"Tokenizer vocabulary updated to {new_vocab_size} tokens")
        return self

    @staticmethod
    def web_lookup(query, engine="google", max_results=5, cache=True, timeout=10):
        """Enhanced web search with caching and fallback engines"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_file = Path(CACHE_DIR) / f"{engine}_{query_hash}.json"
        
        # Check cache
        if cache and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    if 'results' in cached:
                        return cached['results']
            except Exception as e:
                logger.error(f"Cache read error: {e}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "DNT": "1"
        }
        
        snippets = []
        query_encoded = quote(query)
        engines = [engine] + [e for e in ["google", "bing", "ddg"] if e != engine]
        
        for current_engine in engines:
            try:
                if current_engine == "google":
                    url = f"https://www.google.com/search?q={query_encoded}&num={max_results+2}"
                    response = requests.get(url, headers=headers, timeout=timeout)
                    soup = BeautifulSoup(response.text, "html.parser")
                    snippets = [div.get_text(strip=True) for div in soup.select("div.VwiC3b, .MUxGbd")]
                    
                elif current_engine == "bing":
                    url = f"https://www.bing.com/search?q={query_encoded}&count={max_results}"
                    response = requests.get(url, headers=headers, timeout=timeout)
                    soup = BeautifulSoup(response.text, "html.parser")
                    snippets = [p.get_text(strip=True) for p in soup.select("li.b_algo p")]
                    
                elif current_engine == "ddg":
                    url = f"https://html.duckduckgo.com/html/?q={query_encoded}"
                    response = requests.post(url, headers=headers, timeout=timeout)
                    soup = BeautifulSoup(response.text, "html.parser")
                    snippets = [div.get_text(strip=True) for div in soup.select(".result__snippet")]
                
                # Filter and truncate
                snippets = [re.sub(r'\s+', ' ', s)[:500] for s in snippets if s][:max_results]
                if snippets:
                    break
                    
            except Exception as e:
                logger.error(f"Web search error ({current_engine}): {e}")
                continue
        
        # Save to cache
        if cache and snippets:
            try:
                with open(cache_file, 'w') as f:
                    json.dump({'query': query, 'engine': engine, 'results': snippets}, f)
            except Exception as e:
                logger.error(f"Cache write error: {e}")
                
        return snippets

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_beams: int = 1,
        early_stopping: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Enhanced text generation with multiple strategies:
        - Greedy decoding (num_beams=1)
        - Beam search (num_beams>1)
        - Top-k and top-p sampling
        - Repetition penalty
        - KV caching for efficiency
        """
        # Set model to eval mode
        self.eval()
        
        # Handle single sample without batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize generation containers
        generated = input_ids
        past_key_values = None
        
        # Beam search initialization
        if num_beams > 1:
            return self._beam_search(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                early_stopping=early_stopping,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )
        
        # Greedy decoding and sampling
        for step in range(max_length):
            # Create attention mask
            past_length = past_key_values[0][0].size(2) if past_key_values else 0
            attn_mask = self._create_attention_mask(
                generated, 
                past_length=past_length
            )
            
            # Forward pass with past key values
            logits, past_key_values = self(
                generated if past_key_values is None else generated[:, -1:],
                attention_mask=attn_mask,
                past_key_values=past_key_values
            )
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                self._apply_repetition_penalty(next_token_logits, generated, repetition_penalty)
            
            # Apply temperature
            next_token_logits = next_token_logits / max(temperature, 1e-5)
            
            # Apply top-k and top-p filtering
            next_token_logits = self._top_k_top_p_filtering(next_token_logits, top_k, top_p)
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Early stopping if all sequences end with eos_token_id
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        
        return generated

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        num_beams: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        early_stopping: bool,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int]
    ) -> torch.Tensor:
        """Advanced beam search implementation with length normalization"""
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Initialize beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9  # Force first beam
        beam_scores = beam_scores.view(-1)
        
        # Expand input to num_beams copies
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        
        # Initialize sequences
        generated = input_ids
        past_key_values = None
        done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)
        
        for step in range(max_length):
            # Create attention mask
            past_length = past_key_values[0][0].size(2) if past_key_values else 0
            attn_mask = self._create_attention_mask(
                generated, 
                past_length=past_length
            )
            
            # Forward pass
            logits, past_key_values = self(
                generated if past_key_values is None else generated[:, -1:],
                attention_mask=attn_mask,
                past_key_values=past_key_values
            )
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                self._apply_repetition_penalty(next_token_logits, generated, repetition_penalty)
            
            # Apply temperature
            next_token_logits = next_token_logits / max(temperature, 1e-5)
            
            # Apply top-k and top-p filtering
            next_token_logits = self._top_k_top_p_filtering(next_token_logits, top_k, top_p)
            
            # Calculate scores
            next_scores = F.log_softmax(next_token_logits, dim=-1) + beam_scores.unsqueeze(-1)
            
            # Reshape for beam handling
            vocab_size = next_scores.size(-1)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top candidates
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # Prepare for next generation step
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Update sequences and scores
            beam_outputs = []
            new_past_key_values = []
            
            for i in range(batch_size):
                # Select beams for this batch item
                beams = []
                for j in range(2 * num_beams):
                    beam_idx = i * num_beams + next_indices[i, j]
                    token = next_tokens[i, j]
                    score = next_scores[i, j]
                    
                    # Skip done beams except for the first candidate
                    if done[beam_idx] and j > 0:
                        continue
                        
                    beams.append((score, beam_idx, token))
                    
                    if len(beams) == num_beams:
                        break
                
                # Update sequences
                for rank, (score, beam_idx, token) in enumerate(beams):
                    new_seq = torch.cat([
                        generated[beam_idx], 
                        token.unsqueeze(0)
                    ])
                    
                    # Check for EOS
                    is_eos = token == eos_token_id
                    done[i * num_beams + rank] = done[beam_idx] or is_eos
                    
                    # Update score if EOS
                    if is_eos:
                        score = score / (step + 1)  # Length normalization
                        
                    beam_outputs.append(new_seq)
                    beam_scores[i * num_beams + rank] = score
            
            # Reorganize past key values
            generated = torch.stack(beam_outputs)
            
            # Early stopping if all beams are done
            if early_stopping and done.all():
                break
        
        # Select best sequences
        best_sequences = []
        for i in range(batch_size):
            beam_group = generated[i*num_beams:(i+1)*num_beams]
            scores = beam_scores[i*num_beams:(i+1)*num_beams]
            best_idx = scores.argmax()
            best_sequences.append(beam_group[best_idx])
        
        return torch.stack(best_sequences)

    def _apply_repetition_penalty(self, logits, sequences, penalty):
        """Apply repetition penalty to logits"""
        for i, seq in enumerate(sequences):
            unique_tokens, counts = torch.unique(seq, return_counts=True)
            for token, count in zip(unique_tokens, counts):
                if count > 1:  # Only penalize repeated tokens
                    logits[i, token] /= penalty ** count
        return logits

    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0):
        """Filter logits using top-k and top-p (nucleus) sampling"""
        if top_k > 0:
            # Remove all tokens with probability < top_k token
            values, _ = torch.topk(logits, top_k, dim=-1)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_values, torch.tensor(float('-inf')).to(logits.device), logits)
        
        if top_p > 0.0:
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Create mask to remove tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift indices to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create mask to remove tokens
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits

    def augment_input(self, input_text, tokenizer, engine="google", max_context=500, 
                     relevance_threshold=0.3, max_retries=2):
        """Enhanced input augmentation with relevance filtering and fallback"""
        # First try exact match search
        snippets = QuantumTransformer.web_lookup(input_text, engine, max_results=5)
        context = ""
        
        # Filter for relevance
        if snippets:
            # Simple relevance scoring
            relevant_snippets = []
            input_words = input_text.lower().split()[:5]
            for snippet in snippets:
                if any(word in snippet.lower() for word in input_words):
                    relevant_snippets.append(snippet)
            
            context = "\n".join(relevant_snippets)[:max_context]
        
        # Fallback to broader search if no relevant results
        if not context and max_retries > 0:
            # Try with broader query
            broader_query = ' '.join(input_text.split()[:3])
            return self.augment_input(
                broader_query, tokenizer, engine, max_context, 
                relevance_threshold, max_retries-1
            )
        
        if context:
            # Tokenize context to update vocabulary
            context_tokens = tokenizer.tokenize(context, add_special_tokens=False)
            new_token_count = len(set(context_tokens) - set(range(tokenizer.vocab_size)))
            
            # Expand model vocabulary if needed
            if new_token_count > 0:
                new_size = tokenizer.vocab_size + new_token_count
                self.expand_vocab(new_size, tokenizer)
            
            return f"{input_text}\n\n[CONTEXT]\n{context}"
        
        return input_text

    def save_pretrained(self, path):
        """Save model with configuration"""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.bin")
        
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "heads": self.heads,
            "layers": self.layers,
            "dropout": self.dropout_rate,
            "max_len": self.max_len,
            "use_quantum_ffn": any(
                isinstance(layer, QuantumTransformerLayer) 
                and layer.use_quantum_ffn 
                for layer in self.encoder_layers
            ),
            "use_rotary_emb": self.use_rotary_emb
        }
        with open(Path(path) / "config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path):
        """Load model from pretrained"""
        with open(Path(path) / "config.json", "r") as f:
            config = json.load(f)
        
        model = cls(**config)
        model.load_state_dict(torch.load(Path(path) / "model.bin", map_location=model.device))
        return model

class MiniTransformer(QuantumTransformer):
    """Enhanced lightweight version of QuantumTransformer"""
    def __init__(self, vocab_size=4096, dim=256, heads=4, layers=4, max_len=256, 
                 dropout=0.1, use_quantum_ffn=False, use_rotary_emb=True):
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            max_len=max_len,
            use_quantum_ffn=use_quantum_ffn,
            use_rotary_emb=use_rotary_emb
        )
