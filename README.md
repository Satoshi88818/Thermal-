Block-Sparse Discrete Thermal Model v13

Thermalv13 is a production-grade generative image model combining energy-based modeling, score-based diffusion, and a learned spatial thermal field (τ) that adapts noise injection per image region. The model is trained on CIFAR-10 at 64×64 resolution and is designed with extensibility toward higher-resolution and domain-specific applications.

Table of Contents

Architecture 

Overview

Sub-modules

Thermal Field τ

Diffusion Schedule

Loss Functions

Requirements

Installation

Deployment 

Training

Resuming Training

Inference / Sampling

Running Tests

Configuration

Distributed Training

Potential Use Cases

Current Limitations

Suggested Enhancements for Scaling

Changelog: v12 → v13

Architecture

Overview

The model decomposes images into a two-level block hierarchy — fine blocks (8×8 pixels) and coarse blocks (16×16 pixels) — and learns a separate energy function for each level. A learned thermal field τ determines how much noise each spatial region receives during forward diffusion, allowing the model to focus generative capacity where the image is most complex.

At inference, the model uses ancestral DDPM sampling with the Evolutionary Filtration Loop (EFL) buffer active, providing temporal memory across denoising steps.

Input Image (64×64×3) │ ├──► CondProcessor (U-Net) ──────────────────────────────────────┐ │ │ │ │ [B, 128, 8, 8] conditioning features │ │ │ ├──► Extract Fine Blocks [B, 64, 192] ──► DilatedBlockChain ──► τ, mu, logvar, chain_ctx │ │ │ EFLBuffer (per-sample EMA) │ ├──► Extract Coarse Blocks [B, 16, 768] ──► BlockSparseEnergy (coarse) │ │ └──► Fine Blocks + chain_ctx + cond_feat ──► BlockSparseEnergy (fine) │ E = (E_coarse + E_fine) × 1/√τ × temp_scale │ Score = −∇_{x} E(x, t) 

Sub-modules

CondProcessor

A lightweight U-Net that encodes a conditioning RGB image (64×64) into an 8×8 spatial feature map at 128 channels.

Four encoder stages with stride-2 convolutions: 3→16→32→64→128 channels

Three decoder stages with transposed convolutions and additive skip connections

GroupNorm (8 groups) throughout — AMP-safe, small-batch stable, identical at train/eval time

Output: [B, 128, 8, 8] → spatially aligned with the fine block grid

DilatedBlockChain

A shift-equivariant 2D dilated convolutional chain that processes the fine block grid and produces the variational thermal field.

Blocks are reshaped into a 2D spatial grid: [B, H_blocks, W_blocks, features]

Projected to chain_hidden=256 dimensions

Six GRU-style gated steps at dilations [1, 2, 4, 8, 16, 32] — providing a large receptive field without parameter growth

Each step: h = LayerNorm((1 − gate) × h + gate × tanh(update))

Outputs: per-block hidden states chain_ctx, plus variational τ parameters mu and logvar

During training: τ = mu + exp(0.5 × logvar) × ε (reparameterization trick)

During inference: τ = mu

BlockSparseEnergy

A per-block energy MLP with a skip connection, run independently on every block.

Architecture: 4-layer main branch (hidden dim 2048, LayerNorm + ReLU) + 2-layer skip branch

Fine blocks receive concatenated conditioning: [x_block ‖ cond_features ‖ chain_ctx]

Coarse blocks receive only [x_block]

Energy convention: lower energy = more likely state

Energy formula per block: E = −(h_main + 0.3×h_skip).sum() − bias·x − t_bias·x

Blocks aggregated as: E_total = Σ E_block × (1/√τ) — precision-weighted, not temperature-divided

EFLBuffer (Evolutionary Filtration Loop)

A per-sample EMA buffer of chain hidden states, providing cross-step temporal memory.

Stores [B, N_blocks, hidden_dim] — one state vector per sample, preventing cross-sample information leakage

Learned sigmoid gate controls how much the current input vs. stored memory contributes

EMA state is always stop-gradient — no temporal loops in backpropagation

Must be reset(B) at the start of each new batch or sampling call

Inactive during DSM warmup; activated after efl_warmup_epochs

ModelEMA

A Polyak-averaged shadow copy of model weights for more stable evaluation and sampling.

Initialised at ema_start_epoch=5 to avoid averaging over early noisy weights

ema.context(model) is a context manager that temporarily swaps live weights with the shadow for eval/sampling, then restores them

DDP-aware: unwraps model.module when present

Thermal Field τ

τ is the central innovation. It is a per-block learned temperature that controls:

Forward noise injection: noise ~ N(0, τ_map) — hotter blocks receive proportionally more noise

Energy precision weighting: block energies are scaled by 1/√τ — uncertain blocks contribute less to the total energy

Bilinear interpolation: τ is upsampled from block-level to pixel-level for spatially smooth noise maps

τ is constrained by three mechanisms:

softplus(mu_raw) + 0.1 ensures positivity

Ceiling penalty: squared hinge above tau_ceiling=8.0

Smoothness regularization: penalizes large differences between adjacent block τ values

Diffusion Schedule

Cosine beta schedule (Nichol & Dhariwal, 2021) over T=100 steps:

αbar_t = cos((t/T) × π/2)² / cos(0)² β_t = 1 − αbar_t / αbar_{t−1} β_t clipped to [0.0001, 0.9999] 

Forward process: x_t = √αbar_t × x_0 + √(1−αbar_t) × noise × √τ_map

Loss Functions

DSM Loss (Warmup — EFL disabled)

Used for the first efl_warmup_epochs=5 epochs:

L_dsm = MSE(score, −noise / √(1−αbar_t)) + λ_smooth_dsm × smooth(mu_τ) + τ_penalty × ceiling(mu_τ) 

General Coherence Loss (EFL active)

Four terms, with EFL-specific terms linearly ramped in over the warmup period:

TermFormulaWeightDSM anchorMSE(score, target)λ_dsm = 0.5Self-consistencyMSE(mu_n, mu_{n+1}.detach())λ_consist = 0.3 × rampEntropic balance−logvar.mean() + λ_smooth_entropy × smooth(mu)λ_entropy = 0.2 × rampτ ceilingmean(relu(τ − ceiling)²)τ_penalty = 0.01 

Requirements

Python & CUDA

DependencyMinimum VersionNotesPython3.9+PyTorch2.0+Required for torch.compileCUDA11.8+Optional; CPU training supportedtorchvision0.15+CIFAR-10 data loadingnumpy1.23+matplotlib3.5+Sample grid saving 

Optional

DependencyPurposewandbExperiment trackingMulti-GPU environmentDistributed training via DDP 

Hardware Recommendations

ModeGPU VRAMNotesSmoke test / unit tests4 GBCPU also worksTraining (batch=16)8 GBDefault configTraining (batch=32)16 GBRecommended for stable τ estimatesDistributed (2× GPU)8 GB × 2NCCL backend 

Installation

# Clone the repository git clone https://github.com/your-org/thermal-dtm-v13.git cd thermal-dtm-v13 # Create and activate a virtual environment python -m venv venv source venv/bin/activate # Windows: venv\Scripts\activate # Install core dependencies pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 # Install remaining dependencies pip install numpy matplotlib # Optional: experiment tracking pip install wandb 

Deployment

Training

# Default training run (CIFAR-10, 50 epochs, auto device) python Thermalv13.py --mode train # With custom settings python Thermalv13.py --mode train \ --epochs 100 \ --batch-size 32 \ --output-dir ./runs/experiment_01 \ --seed 123 # With W&B logging python Thermalv13.py --mode train \ --wandb \ --wandb-project my-thermal-project # Disable torch.compile (e.g. for debugging) python Thermalv13.py --mode train --no-compile # CPU-only (disables AMP automatically) python Thermalv13.py --mode train --no-amp 

Resuming Training

python Thermalv13.py --mode train \ --resume ./runs/experiment_01/ckpt_epoch0020.pt \ --output-dir ./runs/experiment_01 

The checkpoint stores model weights, optimizer state, EMA shadow, RNG state (CPU + CUDA), epoch number, and config. Training resumes from the next epoch automatically.

Inference / Sampling

Sampling is integrated into the training loop (every sample_interval=5 epochs by default). For standalone inference, use the model API directly:

import torch from Thermalv13 import ThermalConfig, BlockSparseDTM, ModelEMA, load_checkpoint # Load config and model cfg = ThermalConfig.load("./runs/experiment_01/config.json") model = BlockSparseDTM(cfg).to(cfg.device) ema = ModelEMA(model, decay=cfg.ema_decay) optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr) load_checkpoint(model, optimizer, "./runs/experiment_01/ckpt_best.pt", cfg, ema=ema) # Generate samples using EMA weights model.eval() with ema.context(model): samples = model.sample(n_samples=16, verbose=True) # [16, 3, 64, 64] # Save grid from Thermalv13 import save_image_grid save_image_grid(samples, "generated_samples.png", nrow=4) 

Running Tests

# Full smoke test suite (recommended before any training run) python Thermalv13.py --mode smoke # Individual tests python Thermalv13.py --mode rftest # Dilated chain receptive field python Thermalv13.py --mode efltest # EFL per-sample isolation python Thermalv13.py --mode ckpttest # Checkpoint round-trip python Thermalv13.py --mode ematest # EMA shadow / context swap 

All five tests run automatically before training in --mode train.

Configuration

All hyperparameters are stored in ThermalConfig and can be overridden via a JSON file:

{ "img_size": 64, "channels": 3, "fine_block_size": 8, "coarse_block_size": 16, "chain_hidden": 256, "chain_dilations": [1, 2, 4, 8, 16, 32], "energy_hidden": 2048, "t_steps": 100, "batch_size": 16, "lr": 2e-4, "weight_decay": 1e-4, "num_epochs": 50, "ema_decay": 0.9999, "ema_start_epoch": 5, "efl_warmup_epochs": 5, "lambda_dsm": 0.5, "lambda_consist": 0.3, "lambda_entropy": 0.2, "tau_ceiling": 8.0, "tau_penalty_weight": 0.01 } python Thermalv13.py --mode train --config my_config.json 

Distributed Training

# 2-GPU training on a single node torchrun --nproc_per_node=2 Thermalv13.py --mode train # 4-GPU across 2 nodes torchrun \ --nnodes=2 \ --nproc_per_node=2 \ --rdzv_backend=c10d \ --rdzv_endpoint=master_node:29500 \ Thermalv13.py --mode train 

DDP uses NCCL on CUDA and Gloo on CPU. Only rank 0 writes checkpoints, sample grids, and logs.

Potential Use Cases

Research & Academic

Content-aware diffusion research. The τ field is a novel mechanism for spatially adapting noise in diffusion models. It provides a principled way to study how generative capacity is allocated across spatial regions, with implications for sample quality, training stability, and interpretability.

Energy-based / score-based model unification. The architecture bridges EBMs and diffusion models by computing scores as gradients of an energy function. This makes it a useful experimental platform for studying that relationship.

Uncertainty quantification. The variational τ (mu + logvar per block) provides an explicit per-region uncertainty estimate that could be studied independently of the generative objective.

Applied Generative Applications

Conditional image synthesis. The cond_image pathway is fully wired — providing a different image as the conditioning input enables image-to-image translation with minimal modification. Possible targets include super-resolution, inpainting, and style transfer.

Spatially-prioritized generation. Because τ concentrates generative effort on complex regions, the model is naturally suited to domains where certain areas require high fidelity and others do not — medical imaging (lesion regions), satellite imagery (building/road structures), or face generation (eyes, hair).

Anomaly detection. The energy function assigns low values to in-distribution data. High energy at inference time is a signal that an input is anomalous — applicable to manufacturing defect detection, medical screening, and fraud detection on image data.

Industrial & Scientific

Scientific simulation. Spatially varying uncertainty directly mirrors physical systems where some regions evolve faster or more chaotically — fluid dynamics, material stress, atmospheric modeling at patch level. The τ field could be adapted to represent local simulation confidence.

Compression artifact prediction. The block-sparse structure is architecturally aligned with JPEG-style block coding. The model could be repurposed to predict perceptual quality or blocking artifacts per spatial region, useful in video/image compression pipelines.

Curriculum signal for supervised learning. The τ map identifies which image regions the model finds complex. This could be extracted and used to guide hard-example mining, attention weighting, or data augmentation strategies in supervised downstream tasks.

World models for reinforcement learning. The EFL buffer's temporal memory combined with block-sparse spatial decomposition resembles what is needed for environment modeling in RL — decomposed, memory-augmented, and spatially structured.

Current Limitations

Resolution. The model is trained and validated at 64×64. Scaling to 256×256 increases the fine block count from 64 to 1,024, which stresses the EFL buffer, energy MLP batch dimensions, and autograd memory simultaneously.

autograd.grad score computation. Every forward pass retains the full computation graph to compute −∇E. This doubles memory versus a standard forward pass and is the primary bottleneck for scaling to larger images or batch sizes.

Energy MLP size. energy_hidden=2048 across 1,024 fine blocks at 256×256 becomes expensive. The MLP runs independently per block, so compute scales linearly with N — but that linear scaling is steep at the current hidden dimension.

Single-dataset training. The data pipeline is hardcoded to CIFAR-10. Adapting to other datasets requires modifying get_cifar10_loaders and potentially the CondProcessor input normalization.

No data augmentation. Only resize and normalize are applied. For a generative model this is workable, but it limits diversity and robustness of learned representations.

EMA initialised mid-training. EMA starts at ema_start_epoch=5. Checkpoints saved before this point have no EMA shadow, which can cause resume failures if the checkpoint predates EMA initialization.

Sampling speed. With T=100 denoising steps and autograd score computation at each step, sampling 16 images is slow. There is no DDIM or other accelerated sampler implemented.

No FID/IS evaluation. Sample quality is assessed visually via saved grids. There is no automated quantitative evaluation integrated into the training loop.

Suggested Enhancements for Scaling

Priority 1 — Amortize Score Computation

Replace the autograd.grad score with a learned score network trained to predict −∇E directly:

class ScoreNetwork(nn.Module): """U-Net that predicts score given (x_t, t), supervised by autograd score.""" ... # During training: generate autograd scores as targets, distill into ScoreNetwork # During inference: use ScoreNetwork directly — no graph retention needed 

This is the single highest-leverage change, unlocking larger batch sizes and faster sampling.

Priority 2 — Increase Block Size

Switching from 8×8 to 16×16 fine blocks at 256×256 reduces N from 1,024 to 256 — a 4× reduction in EFL buffer size, energy MLP batch dimension, and chain grid size. Sample quality impact should be validated on a held-out set before committing.

Priority 3 — Cascaded Generation

Leverage the existing cond_image path for hierarchical synthesis:

Stage 1: Train at 64×64 (current model) Stage 2: Condition on 64×64 output, generate residuals at 256×256 Stage 3: Condition on 256×256 output, generate residuals at 1024×1024 

Each stage is a separate BlockSparseDTM instance. Stage 2+ only models high-frequency residuals, keeping the energy function tractable.

Priority 4 — Gradient Checkpointing

Apply torch.utils.checkpoint to the dilated chain and energy MLP to trade recomputation for memory, without changing the model architecture:

from torch.utils.checkpoint import checkpoint # In DilatedBlockChain.forward: for i, d in enumerate(self.dilations): h_grid = checkpoint(self._gated_step, h_grid, i) 

Expected memory reduction: 40–60%, enabling roughly 2× larger batches at the same VRAM.

Priority 5 — Slim the Energy MLP

Replace the monolithic 2048-hidden MLP with a shared-trunk + block-specific bias design:

class SharedTrunkEnergy(nn.Module): """One shared MLP for all blocks + small per-block bias vectors.""" def __init__(self, features, n_blocks, hidden): self.trunk = nn.Sequential(...) # shared self.biases = nn.Parameter(torch.zeros(n_blocks, features)) # per-block 

This reduces parameters significantly while retaining per-block specialization via learned offsets.

Priority 6 — Accelerated Sampling

Implement DDIM sampling for 10–20× speedup at inference:

@torch.no_grad() def sample_ddim(self, n_samples, n_steps=20, eta=0.0, ...): """DDIM sampler — deterministic (eta=0) or stochastic (eta>0).""" timesteps = torch.linspace(self.T - 1, 0, n_steps).long() for t_cur, t_prev in zip(timesteps[:-1], timesteps[1:]): score = self.compute_score(x, t_tensor, ...) x0_pred = ... # DDIM update x = ... return x 

Priority 7 — Block-Parallel Distributed Training

The block-sparse structure allows partitioning blocks across GPUs:

GPU 0 owns blocks 0..N/2, GPU 1 owns blocks N/2..N

The dilated chain requires all-reduce at each dilation boundary

Energy computation is embarrassingly parallel across blocks

This is more complex than standard DDP but architecturally natural given the block-sparse design.

Priority 8 — Dataset Generalization

Abstract the data pipeline to support arbitrary image datasets:

def get_image_loaders(cfg, dataset_path, distributed=False): """Generic loader for any torchvision-compatible image folder dataset.""" transform = transforms.Compose([ transforms.Resize(cfg.img_size), transforms.CenterCrop(cfg.img_size), transforms.ToTensor(), transforms.Normalize(mean, std), ]) dataset = datasets.ImageFolder(dataset_path, transform=transform) ... 

Priority 9 — Quantitative Evaluation

Integrate FID scoring into the validation loop:

# Using torch-fidelity or clean-fid from cleanfid import fid fid_score = fid.compute_fid( gen_folder, dataset_folder, mode="clean", device=cfg.device ) self.tracker.log({"eval/fid": fid_score}, step=epoch) 

This provides an objective signal for comparing architectural variants, loss weights, and scaling decisions.

Changelog: v12 → v13

Componentv12v13Energy scalingDivision by τMultiplicative 1/√τ (precision weighting)τ resolutionPer-block scalarPer-pixel via bilinear interpolationτ parameterizationDeterministicVariational (mu + logvar, reparameterization trick)Entropy regularization−log(var(τ))ELBO-style −logvar.mean()Spatial contextFixed grid + Hann windowShift-equivariant 2D dilated convolutional chainAnti-aliasingHann windowRemoved (handled by convolutional chain)Consistency / smoothness / ceiling targetsτ directlymu_τEFL per-sample isolationShared EMA vector [1, N, H]Per-sample [B, N, H] 

Block-Sparse DTM v13 — Production Grade

