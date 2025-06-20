import os
from dotenv import load_dotenv

load_dotenv()  # This loads .env into environment variables

MODEL_PATH = os.getenv("MODEL_PATH")
USE_CUDA = os.getenv("USE_CUDA", "False") == "True"
import time
import torch
import json
import platform
from torch import nn, optim
from tokenizer.tokenizer import NeuroQuantumTokenizer
from model.transformer import MiniTransformer
from config import *
from utils import (
    DEVICE_MANAGER, WEB_UTILS, DataUtils, ModelUtils,
    PerfUtils, SecurityUtils, DistributedUtils
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

use_mixed_precision = use_cuda  # AMP only on GPU
scaler = GradScaler(enabled=use_mixed_precision)

# ========== ENVIRONMENT SETUP ==========
print("\n" + "="*60)
print("⚡ QUANTUM TRANSFORMER TRAINING SYSTEM")
print("="*60)
print(f"• Neural Core ID: {hash(time.time()):x}")
print(f"• Quantum Dimensions: {MODEL_DIM}D")
print(f"• Neural Layers: {NEURAL_LAYERS}")
print(f"• Attention Heads: {NUM_HEADS}")
print(f"• Context Window: {SEQ_LENGTH} tokens")
print(f"• Device: {DEVICE_MANAGER.device}")
print(f"• Distributed: {'ENABLED' if DISTRIBUTED else 'DISABLED'}")
print(f"• Precision: {'Mixed' if MIXED_PRECISION else 'Full'}")
print(f"• Log Path: {LOG_PATH}")
print(f"• Model Save Path: {MODEL_SAVE_PATH}")
print("-" * 60 + "\n")

# ========== DISTRIBUTED SETUP ==========
if DISTRIBUTED:
    print("🚀 Initializing distributed training...")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    DEVICE = DistributedUtils.setup_ddp(local_rank, world_size)
    is_main_process = DistributedUtils.is_main_process()
    print(f"• Process {local_rank+1}/{world_size} | Device: {DEVICE}")
else:
    is_main_process = True
    DEVICE = DEVICE_MANAGER.device

# ========== TOKENIZER INITIALIZATION ==========
tokenizer = NeuroQuantumTokenizer(mode='char')
if is_main_process:
    print(f"\n🧩 Initializing tokenizer...")
    # Correctly handle vocab_size as either property or method
    if callable(tokenizer.vocab_size):
        vocab_size = tokenizer.vocab_size()
    else:
        vocab_size = tokenizer.vocab_size
    print(f"• Vocabulary size: {vocab_size}")
    print(f"• Special tokens: {tokenizer.special_tokens}")

# ========== DATASET MANAGEMENT SYSTEM ==========
class QuantumDataset:
    """Advanced dataset manager with dynamic expansion capabilities"""
    def __init__(self, base_paths, tokenizer, seq_length=SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.base_paths = base_paths
        self.data = []
        self.cache_file = "dataset_cache.bin"
        self._load_base_data()
        
    def _load_base_data(self):
        """Load and tokenize base datasets"""
        if is_main_process:
            print(f"\n📂 Loading base datasets...")
            
        for path in self.base_paths:
            if not os.path.exists(path):
                if is_main_process:
                    print(f"[⚠️] Dataset not found: {path}")
                continue
                
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
                tokens = self.tokenizer.tokenize(text)
                self.data.extend(tokens)
                
        if is_main_process:
            print(f"• Base tokens: {len(self.data):,}")
    
    def _web_expansion(self, query, max_tokens=50000):
        """Expand dataset with web content"""
        if is_main_process:
            print(f"\n🌐 Web expansion: '{query}'")
            
        snippets = WEB_UTILS.search_web(query, max_results=WEB_SEARCH_RESULTS)
        new_text = "\n".join(snippets)
        new_tokens = self.tokenizer.tokenize(new_text)
        
        if len(new_tokens) > max_tokens:
            new_tokens = new_tokens[:max_tokens]
            
        self.data.extend(new_tokens)
        
        if is_main_process:
            print(f"• Added {len(new_tokens):,} web tokens")
        return len(new_tokens)
    
    def _darknet_expansion(self, query, max_tokens=30000):
        """Simulate darknet data acquisition (placeholder)"""
        if is_main_process:
            print(f"\n🌑 Darknet expansion: '{query}'")
            
        # In a real implementation, this would connect to darknet sources
        # For demonstration, we'll generate synthetic data
        synthetic_text = f"[DARKNET] {query} " * 500
        new_tokens = self.tokenizer.tokenize(synthetic_text)
        
        if len(new_tokens) > max_tokens:
            new_tokens = new_tokens[:max_tokens]
            
        self.data.extend(new_tokens)
        
        if is_main_process:
            print(f"• Added {len(new_tokens):,} synthetic darknet tokens")
        return len(new_tokens)
    
    def _feepnet_expansion(self, concept, max_tokens=40000):
        """Simulate feepnet knowledge integration (placeholder)"""
        if is_main_process:
            print(f"\n🧠 Feepnet expansion: '{concept}'")
            
        # Simulate conceptual knowledge integration
        conceptual_text = (
            f"Concept: {concept}\n\n"
            f"Definition: In quantum information theory, {concept} refers to the fundamental "
            "principle that allows quantum systems to exist in superposition states and "
            "exhibit entanglement properties that enable exponential computational advantages "
            "over classical systems when properly harnessed through quantum algorithms."
        ) * 100
        new_tokens = self.tokenizer.tokenize(conceptual_text)
        
        if len(new_tokens) > max_tokens:
            new_tokens = new_tokens[:max_tokens]
            
        self.data.extend(new_tokens)
        
        if is_main_process:
            print(f"• Added {len(new_tokens):,} conceptual tokens")
        return len(new_tokens)
    
    def dynamic_expansion(self, strategy):
        """Apply dataset expansion strategy"""
        total_added = 0
        for item in strategy:
            source = item['source']
            query = item['query']
            
            if source == 'web':
                total_added += self._web_expansion(query)
            elif source == 'darknet':
                total_added += self._darknet_expansion(query)
            elif source == 'feepnet':
                total_added += self._feepnet_expansion(query)
        
        if is_main_process:
            print(f"\n🚀 Total tokens added: {total_added:,}")
            print(f"📊 New dataset size: {len(self.data):,} tokens")
        return total_added
    
    def get_tensors(self):
        """Convert dataset to input/target tensors"""
        # Ensure we have at least 2 tokens
        if len(self.data) < 2:
            raise ValueError("Dataset is too small. Need at least 2 tokens.")
            
        input_ids = torch.tensor(self.data[:-1], dtype=torch.long)
        target_ids = torch.tensor(self.data[1:], dtype=torch.long)
        return input_ids, target_ids
    
    def save_cache(self):
        """Save dataset to cache"""
        torch.save({
            'data': self.data,
            'tokenizer_state': self.tokenizer.get_state()
        }, self.cache_file)
        if is_main_process:
            print(f"💾 Dataset cache saved: {self.cache_file}")
    
    def load_cache(self):
        """Load dataset from cache"""
        if os.path.exists(self.cache_file):
            cache = torch.load(self.cache_file, map_location='cpu')
            self.data = cache['data']
            self.tokenizer.set_state(cache['tokenizer_state'])
            if is_main_process:
                print(f"♻️ Loaded dataset cache: {len(self.data):,} tokens")
            return True
        return False

# ========== DATASET INITIALIZATION ==========
dataset = QuantumDataset(
    base_paths=["data/sample.txt", "data/additional.txt"],
    tokenizer=tokenizer
)

# Load from cache or initialize
if not dataset.load_cache():
    # Dynamic dataset expansion strategy
    expansion_strategy = [
        {'source': 'web', 'query': 'quantum machine learning research papers'},
        {'source': 'darknet', 'query': 'quantum cryptography protocols'},
        {'source': 'feepnet', 'query': 'quantum entanglement'},
        {'source': 'web', 'query': 'transformer architectures in AI'},
        {'source': 'feepnet', 'query': 'quantum neural networks'}
    ]
    dataset.dynamic_expansion(expansion_strategy)
    dataset.save_cache()

try:
    input_ids, target_ids = dataset.get_tensors()
    if is_main_process:
        print(f"• Dataset tokens: {len(dataset.data):,}")
except ValueError as e:
    print(f"❌ Dataset error: {str(e)}")
    exit(1)

# Get vocab_size properly
if callable(tokenizer.vocab_size):
    vocab_size = tokenizer.vocab_size()
else:
    vocab_size = tokenizer.vocab_size

# ========== MODEL INITIALIZATION ==========
if is_main_process:
    print("\n🧠 Initializing Quantum Transformer Model...")

model = MiniTransformer(
    vocab_size=vocab_size,
    dim=MODEL_DIM,
    heads=NUM_HEADS,
    layers=NEURAL_LAYERS,
    max_len=SEQ_LENGTH,
    use_quantum_ffn=QUANTUM_FFN,
    use_rotary_emb=ROTARY_EMB
).to(DEVICE)

if DISTRIBUTED:
    model = DistributedUtils.ddp_model(model, DEVICE)

if is_main_process:
    print(ModelUtils.model_summary(model, depth=3))
    print(f"• Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"• FLOPs: {ModelUtils.calculate_flops(model, (1, SEQ_LENGTH)):,}")

# ========== TRAINING CONFIGURATION ==========
# Calculate total steps
num_samples = len(input_ids)
num_batches = num_samples // (BATCH_SIZE * SEQ_LENGTH)
steps_per_epoch = (num_batches + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
TOTAL_STEPS = steps_per_epoch * EPOCHS

if is_main_process:
    print(f"\n⚙️ Training Configuration:")
    print(f"• Samples: {num_samples:,}")
    print(f"• Batches per epoch: {num_batches:,}")
    print(f"• Steps per epoch: {steps_per_epoch:,}")
    print(f"• Total steps: {TOTAL_STEPS:,}")

# ========== OPTIMIZATION SETUP ==========
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.98)
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LEARNING_RATE,
    total_steps=TOTAL_STEPS,
    pct_start=WARMUP_PCT,
    anneal_strategy='linear'
)

# Loss function with label smoothing
pad_token_id = tokenizer.stoi.get("<|pad|>", tokenizer.unk_token_id)
loss_fn = nn.CrossEntropyLoss(
    ignore_index=pad_token_id,
    label_smoothing=LABEL_SMOOTHING
)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

# ========== CHECKPOINT SYSTEM ==========
def save_checkpoint(epoch, step, loss, is_best=False):
    """Enhanced checkpoint saving with metadata"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state': model.module.state_dict() if DISTRIBUTED else model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict() if MIXED_PRECISION else None,
        'loss': loss,
        'config': {
            'model_dim': MODEL_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'seq_length': SEQ_LENGTH,
            'vocab_size': vocab_size
        },
        'tokenizer_state': tokenizer.get_state(),
        'timestamp': time.time()
    }
    
    # Save main checkpoint
    torch.save(checkpoint, MODEL_SAVE_PATH)
    
    # Save best checkpoint separately
    if is_best:
        best_path = MODEL_SAVE_PATH.replace(".pt", "_best.pt")
        torch.save(checkpoint, best_path)
    
    if is_main_process:
        status = "BEST" if is_best else "regular"
        print(f"💾 Saved {status} checkpoint [Epoch {epoch}, Step {step}]")

def load_checkpoint():
    """Enhanced checkpoint loading with compatibility checks"""
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            
            # Model loading with size matching
            model_state = checkpoint['model_state']
            current_state = model.module.state_dict() if DISTRIBUTED else model.state_dict()
            
            # Handle vocabulary expansion
            if current_state['embedding.weight'].size(0) != model_state['embedding.weight'].size(0):
                if is_main_process:
                    print("🔍 Vocabulary size mismatch detected - expanding model...")
                model.expand_vocab(model_state['embedding.weight'].size(0), tokenizer)
            
            # Load state dict with strict=False to handle architecture changes
            load_result = model.load_state_dict(model_state, strict=False)
            if is_main_process:
                if load_result.missing_keys:
                    print(f"[⚠️] Missing keys: {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    print(f"[⚠️] Unexpected keys: {load_result.unexpected_keys}")
            
            # Load other states
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            if MIXED_PRECISION and checkpoint.get('scaler_state'):
                scaler.load_state_dict(checkpoint['scaler_state'])
            
            tokenizer.set_state(checkpoint['tokenizer_state'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            start_step = checkpoint.get('step', 0) + 1
            best_loss = checkpoint.get('loss', float('inf'))
            
            if is_main_process:
                print(f"[✅] Loaded checkpoint from epoch {start_epoch-1}, step {start_step-1}")
            return start_epoch, start_step, best_loss
        except Exception as e:
            if is_main_process:
                print(f"[⚠️] Failed to load checkpoint: {str(e)}")
    return 1, 0, float('inf')

# Load checkpoint if available
start_epoch, start_step, best_loss = load_checkpoint()

# ========== BATCH GENERATOR ==========
def get_batches(input_ids, target_ids, batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH):
    """Advanced batch generator with dynamic batching"""
    num_samples = len(input_ids)
    num_batches = num_samples // (batch_size * seq_length)
    
    # Create random indices if shuffling
    if SHUFFLE_DATA:
        indices = torch.randperm(num_samples)
    else:
        indices = torch.arange(num_samples)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size * seq_length
        end_idx = start_idx + batch_size * seq_length
        
        # Select batch indices
        batch_indices = indices[start_idx:end_idx]
        
        # Create batch tensors
        x_batch = input_ids[batch_indices].view(batch_size, seq_length)
        y_batch = target_ids[batch_indices].view(batch_size, seq_length)
        
        yield x_batch.to(DEVICE), y_batch.to(DEVICE)

# ========== TRAINING UTILITIES ==========
def generate_sample(prompt, max_length=100, temperature=0.8):
    """Generate sample text during training"""
    model.eval()
    with torch.no_grad():
        input_seq = tokenizer.tokenize(prompt, add_special_tokens=False)
        if len(input_seq) > SEQ_LENGTH:
            input_seq = input_seq[:SEQ_LENGTH]
        
        input_tensor = torch.tensor([input_seq], dtype=torch.long, device=DEVICE)
        
        # Generate with enhanced sampling
        output = model.generate(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY
        )
        decoded = tokenizer.decode(output[0].cpu().numpy().tolist())
    
    model.train()
    return decoded

def log_performance(epoch, step, loss, lr, tokens_sec):
    """Log training performance to file and console"""
    log_entry = (f"Epoch {epoch:03} | Step {step:06} | Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | Speed: {tokens_sec/1000:.1f}k tok/s")
    
    if is_main_process:
        with open(LOG_PATH, "a") as log_file:
            log_file.write(log_entry + "\n")
        
        # Print with color coding based on loss
        if loss < 1.0:
            color_code = "\033[92m"  # Green
        elif loss < 2.0:
            color_code = "\033[93m"  # Yellow
        else:
            color_code = "\033[91m"  # Red
            
        print(color_code + log_entry + "\033[0m")

# ========== TRAINING LOOP ==========
if is_main_process:
    print(f"\n🚀 Starting training session | Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")
    print(f"• Total steps: {TOTAL_STEPS:,} | Warmup: {WARMUP_PCT*100:.0f}%")
    print(f"• Gradient accumulation: {GRAD_ACCUM_STEPS} steps")
    print(f"• Checkpoint every: {SAVE_EVERY} steps")
    print(f"• Dataset size: {len(dataset.data):,} tokens")
    print("-" * 60)

# Create directories if needed
if is_main_process:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Training state
global_step = start_step
best_loss = float('inf')
total_tokens_processed = 0
start_time = time.time()
current_lr = LEARNING_RATE  # Initialize learning rate

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    epoch_tokens = 0
    accumulation_loss = 0.0
    accumulation_steps = 0
    
    # Get data batches
    batches = list(get_batches(input_ids, target_ids))
    num_batches = len(batches)
    
    for batch_idx, (x_batch, y_batch) in enumerate(batches):
        # Mixed precision context
        with autocast(device_type='cuda' if use_cuda else 'cpu', enabled=use_mixed_precision):
            # Forward pass
            outputs = model(x_batch)
            # Reshape for loss calculation
            outputs = outputs.view(-1, vocab_size)
            targets = y_batch.view(-1)
            
            # Calculate loss
            loss = loss_fn(outputs, targets)
            loss = loss / GRAD_ACCUM_STEPS  # Scale loss for accumulation
        
        # Backward pass
        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update accumulation metrics
        accumulation_loss += loss.item() * GRAD_ACCUM_STEPS  # Unscale for logging
        accumulation_steps += 1
        epoch_loss += loss.item() * GRAD_ACCUM_STEPS
        tokens_this_batch = x_batch.numel()
        epoch_tokens += tokens_this_batch
        total_tokens_processed += tokens_this_batch
        
        # Perform optimization step at accumulation boundary or end of batch
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == num_batches:
            # Gradient clipping
            if use_mixed_precision:
                scaler.unscale_(optimizer)
            SecurityUtils.gradient_clipping(model, max_norm=GRAD_CLIP)
            
            # Optimizer step
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            tokens_per_sec = total_tokens_processed / max(elapsed_time, 1e-6)
            avg_loss = accumulation_loss / accumulation_steps
            
            # Log performance
            if global_step % LOG_INTERVAL == 0:
                log_performance(epoch, global_step, avg_loss, current_lr, tokens_per_sec)
            
            # Reset accumulation
            accumulation_loss = 0.0
            accumulation_steps = 0
            global_step += 1
        
        # Save checkpoint
        if global_step % SAVE_EVERY == 0 and is_main_process and global_step > 0:
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            save_checkpoint(epoch, global_step, avg_loss, is_best)
            
            # Generate sample text
            prompt = "Quantum machine learning"
            sample = generate_sample(prompt)
            print(f"\n🧠 Model Output (Step {global_step}):")
            print(f"Prompt: '{prompt}'")
            print(f"Output: '{sample}'\n")
    
    # Epoch summary
    epoch_avg_loss = epoch_loss / num_batches
    epoch_time = (time.time() - start_time) / 60
    epoch_speed = epoch_tokens / (epoch_time * 60)  # Tokens/sec
    
    if is_main_process:
        print(f"\n📊 [Epoch {epoch:03} Summary]")
        print(f"• Avg Loss: {epoch_avg_loss:.4f}")
        print(f"• Time: {epoch_time:.1f} minutes")
        print(f"• Speed: {epoch_speed/1000:.1f}k tokens/sec")
        print(f"• Learning Rate: {current_lr:.2e}")
        print("-" * 50)
        
        # Dynamic dataset expansion at epoch boundaries
        if epoch % DATASET_EXPANSION_INTERVAL == 0:
            expansion_strategy = [
                {'source': 'web', 'query': f'advancements in quantum computing epoch {epoch}'},
                {'source': 'feepnet', 'query': f'quantum neural networks epoch {epoch}'},
                {'source': 'darknet', 'query': f'quantum security protocols epoch {epoch}'}
            ]
            dataset.dynamic_expansion(expansion_strategy)
            dataset.save_cache()
            input_ids, target_ids = dataset.get_tensors()

# Finalize training
if is_main_process:
    total_time = (time.time() - start_time) / 3600  # Convert to hours
    print(f"\n✅ Training completed in {total_time:.2f} hours")
    print(f"📦 Final model saved to {MODEL_SAVE_PATH}")
    
    # Save final tokenizer state
    tokenizer.save("checkpoints/tokenizer_final.json")
    print("🔤 Tokenizer saved: checkpoints/tokenizer_final.json")
    
    # Performance report
    try:
        input_sample = torch.randint(0, vocab_size, (1, SEQ_LENGTH), device=DEVICE)
        perf_report = PerfUtils.benchmark(model, input_sample)
        print("\n⚡ Performance Report:")
        print(f"• Latency: {perf_report['latency_ms']:.2f} ms")
        print(f"• Memory: {perf_report['memory_mb']:.2f} MB")
        print(f"• Throughput: {perf_report['throughput']:.2f} samples/sec")
    except Exception as e:
        print(f"⚠️ Performance benchmarking failed: {str(e)}")

# Cleanup distributed processes
if DISTRIBUTED:
    torch.distributed.destroy_process_group()
