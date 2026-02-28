<p align="center">
  <a href="https://github.com/calisweetleaf/aeron" rel="noopener">
    <img width=200px height=200px src="logo.png" alt="Aeron Project logo">
  </a>
</p>

<h3 align="center">Aeron v4.0.1</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Version](https://img.shields.io/badge/version-4.0.1-blue.svg)]()
![License](https://img.shields.io/badge/License-Sovereign-blueviolet?style=for-the-badge)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Parameters](https://img.shields.io/badge/parameters-3.85B-orange.svg)]()

</div>

---

<p align="center">
  A research-grade, 3.85B-parameter transformer backbone combining SOTA++ architectural primitives with a neural-level Tree of Thought reasoning engine, three integrated memory systems, RLHF alignment infrastructure, and a production-ready LONPT training pipeline. Aeron represents drop four of five in the Project SOTA and its mission to democratize access to SOTA and industry standard advancements and the redistribution of compute. Aeron follows rlhf, neural router and memory system and then drop 3 Project Moonshine, or what is published as [distill-the-flow](https://github.com/calisweetleaf/distill-the-flow)
  Please stay updated on the https://github.com/calisweetleaf/distill-the-flow and now drop four aeron repositories. All things contained in this repository are under the terms of the Somnus SOvereign Anti-Exploitation License, [somnus-license](https://github.com/calisweetleaf/somnus-license)
</p>

---

## Table of Contents

- [Public Release vs Private Development](#public-release-vs-private-development)
- [About](#about)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training](#training)
- [RLHF Suite](#rlhf-suite)
- [Export and Deployment](#export-and-deployment)
- [Consumer Hardware Scaling](#consumer-hardware-scaling)
- [Repository Structure](#repository-structure)
- [Built Using](#built-using)
- [Authors](#authors)
- [License](#license)

---

## **Operation-Sota First Three Drops:**

- Drop One: [Reinforcement-Learning-Full-Pipeline](https://github.com/calisweetleaf/Reinforcement-Learning-Full-Pipeline)

- Drop Two: [SOTA-Runtime-Core](https://github.com/calisweetleaf/SOTA-Runtime-Core)

- Drop Three: [distill-the-flow](https://github.com/calisweetleaf/distill-the-flow)

## How Aeron Fits

- Aeron represents a novel transformer architecture used to spearhead and demo the prior releases utilizing past and planned future tooling to achieve State of the Art. Currently only documentation for both Project Moonshine (distill-the-flow) and Aeron repositories are public but this will change very fast so always be checking.


## Public Release vs Private Development

This repository contains the **public-facing documentation and research framework** for Aeron. Certain core components are maintained privately and are not included in this public release.

### What's Public ✅

- **Architecture Documentation** — Full specifications, model card, and design rationale
- **Research Framework** — RLHF suite, inference optimizations, model merging utilities
- **Training Infrastructure** — Entry points and scaffolding (see `Training` section)
- **Visualization Outputs** — Architecture diagrams and component analysis
- **Tokenization System** —Tokenizer configuration and validation artifacts

### What's Private 🔒

- **Core Model Implementation** — `aeron.py` (the transformer backbone)
- **Tokenizer Runtime** — `tokenizer_mux.py` (tokenization implementation)
- **Training Pipeline Details** — Internal LONPT documentation

For training run details, see `docs/lonpt_full.md`.

---

## About

Aeron is a production-oriented, research-forward transformer backbone scaled to approximately 3.85 billion parameters. The architecture integrates a complete set of modern transformer primitives — Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU feed-forward networks, and RMSNorm pre-normalization — alongside a native Tree of Thought (ToT) reasoning engine, three distinct memory systems, and a comprehensive RLHF alignment suite.

The project is organized as a composable research framework. The core model in `aeron.py` is protected from ad-hoc modification; all new capabilities are introduced through adapters, wrappers, and separate modules that interface with the model's public API. This constraint enforces architectural discipline while allowing the surrounding infrastructure to evolve.

Aeron's training infrastructure includes the LONPT pipeline (Loss-Optimized Neural Processing and Transformation), which achieved a loss reduction from approximately 10.0 to approximately 3.8 on the current dataset, the compressed training system with breath-safety validation, and a full RLHF suite covering PPO, DPO, reward modeling, inference optimization, and model merging.

**Current development status:** The architecture and training infrastructure are fully implemented and structurally verified. No formal public benchmark results are available for the v4.0.1 configuration. Quality claims should be treated as research-stage pending benchmark publication.

---

## Features

### Core Architecture (v4.0.1)

- **Grouped Query Attention (GQA):** 32 query heads, 8 KV heads. Reduces KV-cache memory 4x relative to standard MHA. FlashAttention-2 compatible via `torch.nn.functional.scaled_dot_product_attention`.
- **Rotary Position Embeddings (RoPE):** Applied to all self-attention layers (`rope_theta=500000`). RoPE-only on the text path; absolute sinusoidal PE retained exclusively for the multimodal fusion path.
- **SwiGLU Feed-Forward Networks:** Three-matrix gate-up-down architecture (`SiLU(W_gate @ x) * (W_up @ x)` fed into `W_down`). Replaces standard two-layer FFN.
- **RMSNorm Pre-Normalization (AeronRMSNorm):** Applied before each sublayer throughout encoder and decoder stacks. Eliminates mean-centering overhead for approximately 10-15% wallclock speedup.
- **Weight-Tied Embeddings:** Input token embeddings and output projection matrix are shared, saving approximately 102M parameters at the default configuration.
- **32k Context Window:** `max_position_embeddings=32768` with `rope_theta=500000` for extended-range extrapolation.

### Reasoning Engine

- **Tree of Thought (ToT) Processor:** `num_tot_branches=4` parallel hypothesis generators, cross-branch attention O(N^2) over branch summaries, confidence-scored pruning, dialectical contradiction resolution, quality-weighted merge.
- **AeronInternalScratchpad:** 64 differentiable memory slots with learned key/value parameters. `write()` is gradient-safe (does not mutate global state mid-forward); `commit_write()` persists for inference. Slot metadata includes type, priority, and temporal encoding.
- **AeronScratchpadAttention:** Multi-head attention over scratchpad slots with type/priority/timestamp metadata embeddings.
- **AeronReasoningEngine (Orchestrator):** Complexity gate (`threshold=0.3`) skips reasoning for simple inputs. ToT runs first; scratchpad write follows strictly after ToT returns. Memory bridge connects episodic memory to ToT context.

### Memory Systems (Three Integrated)

| System | Class | Mechanism |
|---|---|---|
| Episodic External Memory | `NeuralMemoryNetwork` | 1000-slot episodic memory, `memory_dim=512` (independent of d_model), feeds ToT via memory_bridge with shape guard |
| Structured Knowledge | `KnowledgeGraphAttention` | Entity/relation embeddings injected into encoder_output before reasoning |
| Continual Learning | `ContinualLearningModule` | EWC-based Fisher consolidation; task embeddings condition ToT branch exploration |

### Additional Advanced Modules

- `UncertaintyQuantification`: Monte Carlo Dropout, Deep Ensembles, Evidential Deep Learning
- `ActiveLearningManager`: BADGE sampling for intelligent annotation selection (training path only)
- `VisionPatchEmbedding` + `MultimodalFusion`: Vision-language cross-attention with configurable patch size

### RLHF Suite (`RLHF/`)

- **`rlhf.py`:** Full RLHF pipeline — PPO trainer, DPO trainer, reward model training
- **`inference_optimizations.py`:** `OptimizedAttention` (FA2/SDPA), `PagedKVCache`, `SpeculativeDecoder`, `BestOfNSampler`, `MCTSGenerator`, `compile_model`
- **`model_merging.py`:** `ModelMerger` (Task Arithmetic, TIES, SLERP, DARE), `ModelSoup`, `EnsemblePolicy`, `layer_wise_interpolation`

### Training Infrastructure

- **LONPT Pipeline:** Formal graph rewrite engine (Riemannian manifold, ACT/ACTv2), hardware profiler, adaptive control modules. Best known result: ~998MB checkpoint, loss ~3.8.
- **Compressed Training:** Breath-safety validation, sovereignty-preserving quantization (8-32 bit per component type), component-specific compression ratios.
- **Simple and Reference Trainers:** `train_simple.py`, `trainer.py` for rapid iteration.

---

## Architecture

### Forward Pass Data Flow

```
INPUT: input_ids (batch, seq_len)
  |
  +-- Token Embeddings (50k vocab, d_model=2048)
  |   [NO absolute PE on text path -- RoPE handles position inside each attention layer]
  |
  +-- [Optional] Vision Patch Embedding -> Absolute PE -> Multimodal Fusion
  |
  v
ENCODER STACK (32x Pre-Norm Layers)
  |  Each layer: AeronRMSNorm -> GQA Self-Attn (32Q/8KV + RoPE) -> Residual
  |              AeronRMSNorm -> SwiGLU FFN (2048->5461->2048) -> Residual
  |
  +-- encoder_norm (AeronRMSNorm)
  |
  v
ENHANCEMENT PIPELINE (sequential, error-isolated):
  1. KnowledgeGraphAttention    -- structured knowledge injection
  2. NeuralMemoryNetwork        -- 1000-slot episodic memory
  3. ContinualLearningModule    -- EWC consolidation + task conditioning
  4. UncertaintyQuantification  -- evidential deep learning heads
  5. ActiveLearningManager      -- BADGE sampling (training path only)
  |
  v
REASONING ENGINE (AeronReasoningEngine):
  +-- complexity_gate -> skip entirely if complexity < 0.3
  +-- [TREE OF THOUGHT] 4 branches -> cross-branch attention -> critic -> prune ->
  |   contradiction resolution -> quality-weighted merge
  |   (reads NeuralMemoryNetwork via memory_bridge; KG already in encoder_output;
  |    CL task_embedding conditions branch exploration)
  +-- [WRITE TO SCRATCHPAD] strictly after ToT returns
  +-- AeronScratchpadAttention synthesizes across written slots
  |
  v
DECODER STACK (32x Pre-Norm Layers)
  |  Each layer: AeronRMSNorm -> Masked GQA Self-Attn (32Q/8KV + RoPE) -> Residual
  |              AeronRMSNorm -> Cross-Attn (GQA, no RoPE) -> Residual
  |              AeronRMSNorm -> SwiGLU FFN -> Residual
  |
  +-- decoder_norm (AeronRMSNorm)
  |
  v
OUTPUT PROJECTION (weight-tied with token embeddings, bias=False)
  |
  v
OUTPUT: logits (batch, seq_len, vocab_size)
         + tot_branch_scores, scratchpad_stats, reasoning_info,
           knowledge_graph_enhanced, neural_memory_enhanced,
           memory_statistics, uncertainty_estimates
```

### Default Configuration (v4.0.1)

| Parameter | Value | Notes |
|---|---|---|
| `vocab_size` | 50000 | |
| `d_model` | 2048 | Hidden dimension |
| `nhead` | 32 | Query heads (GQA) |
| `num_kv_heads` | 8 | KV heads (GQA) |
| `num_encoder_layers` | 32 | |
| `num_decoder_layers` | 32 | |
| `dim_feedforward` | 8192 | Pre-SwiGLU gate dimension |
| `dropout` | 0.0 | Disabled at 4B scale |
| `rope_theta` | 500000.0 | Extended context RoPE base |
| `max_position_embeddings` | 32768 | 32k context window |
| `num_tot_branches` | 4 | ToT parallel branches |
| `num_scratchpad_slots` | 64 | Differentiable scratchpad slots |
| `max_reasoning_steps` | 3 | Reserved depth hint |
| `reasoning_complexity_threshold` | 0.3 | Below this, skip reasoning |
| **Estimated Parameters** | **~3.85B** | Default config |

### Parameter Budget (Default Config)

| Component | Approximate Parameters |
|---|---|
| Token Embeddings (shared with output) | ~102M |
| Encoder Stack (32 layers) | ~1,376M |
| Decoder Stack (32 layers) | ~2,048M |
| KnowledgeGraphAttention | ~19M |
| NeuralMemoryNetwork | ~15M |
| ContinualLearningModule | ~8M |
| UncertaintyQuantification | ~18M |
| ActiveLearningManager | ~6M |
| AeronReasoningEngine | ~170M |
| **Total** | **~3.85B** |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (required for default 3.85B config; see consumer hardware scaling for reduced configs)
- At minimum 24GB VRAM for default config training; inference may be possible at 16GB with quantization

### Installation

Clone the repository and set up the virtual environment:

```bash
git clone https://github.com/calisweetleaf/aeron
cd aeron

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Verify the environment by running the built-in architecture demo (uses a minimal config to avoid OOM):

```bash
python aeron.py
```

---

## Usage

### Minimal Smoke Test

```python
import torch
from aeron import NeuralNetConfig, TransformerNeuralNetBackbone

# Small config for functional verification (avoids OOM on consumer hardware)
config = NeuralNetConfig(
    d_model=256,
    nhead=4,
    num_kv_heads=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=512,
    vocab_size=50000,
    num_tot_branches=2,
    num_scratchpad_slots=8,
    max_reasoning_steps=1
)

model = TransformerNeuralNetBackbone(config)

ids = torch.randint(0, 50000, (2, 16))
mask = torch.ones(2, 16)

outputs = model(
    input_ids=ids,
    attention_mask=mask,
    decoder_input_ids=ids,
    use_cache=False  # KV-cache not implemented; raises NotImplementedError if True
)

print(outputs['logits'].shape)           # (2, 16, 50000)
print(outputs['tot_branch_scores'].shape) # (2, 2)
print(outputs['scratchpad_stats'])
```

### Full Forward Pass (All Modules Active)

```python
from aeron import NeuralNetConfig, TransformerNeuralNetBackbone
import torch

config = NeuralNetConfig()  # Default 3.85B config
model = TransformerNeuralNetBackbone(config)

outputs = model(
    input_ids=input_ids,                   # (batch, seq_len)
    attention_mask=attention_mask,         # Optional
    vision_inputs=vision_inputs,           # Optional: enables multimodal fusion
    decoder_input_ids=decoder_input_ids,   # Target sequence
    decoder_attention_mask=decoder_mask,   # Optional
    input_entities=entities,              # Optional: enables KG attention
    knowledge_graph=kg_dict,             # Optional: structured knowledge dict
    use_cache=False,                      # KV-cache not implemented
    task_id=None                          # Optional: int for CL task conditioning
)

# Primary output
logits = outputs['logits']                           # (batch, seq_len, vocab_size)

# Reasoning diagnostics
branch_scores = outputs['tot_branch_scores']         # (batch, num_tot_branches) or None
scratchpad = outputs['scratchpad_stats']             # {'used_slots', 'total_slots', 'step_counter'}
reasoning = outputs['reasoning_info']                # Full reasoning diagnostics dict

# Enhancement status flags
kg_enhanced = outputs['knowledge_graph_enhanced']    # bool
mem_enhanced = outputs['neural_memory_enhanced']     # bool
mem_stats = outputs['memory_statistics']             # dict
uncertainty = outputs['uncertainty_estimates']       # dict
```

### Export Mode

Use export mode to disable non-exportable advanced modules before ONNX export. Note: ONNX export produces a logits-only graph suitable for architecture visualization (Netron), not for inference.

```python
model.set_export_mode(True)

from aeron import export_model_to_onnx
export_model_to_onnx(
    model,
    input_ids=input_ids,
    attention_mask=attention_mask,
    decoder_input_ids=decoder_input_ids,
    export_path="exports/model.onnx"
)
```

---

## Training

### LONPT Pipeline (Primary Production Pipeline)

LONPT achieved the best documented training result: loss from ~10 to ~3.8, checkpoint ~998MB.

```bash
python train_lonpt.py
```

LONPT components in `lonpt/`:

| Module | Description |
|---|---|
| `lonpt_graph_transformer.py` | Formal graph rewrite engine (Riemannian manifold, ACT/ACTv2) |
| `lonpt_hardware_profiler.py` / `lonpt_hpf_core.py` | Hardware profiling, HPFLinear precision layers |
| `lonpt_act_transformer.py` | Adaptive computation transformer |
| `lonpt_akap_sequencer.py` | AKAP sequencing module |
| `lonpt_pncec_controller.py` | PNCEC control module |
| `lonpt_integration_controller.py` / `lonpt_core.py` | Control plane and safety rails |
| `aeron_adapter.py` | Bridges Aeron checkpoints into LONPT control flow |

### Compressed Training (Memory-Efficient)

Breath-safety-validated quantization pipeline. Component-specific precision:

| Component | Precision | Compression |
|---|---|---|
| Sovereignty markers | 32-bit | 1.2x |
| Memory networks | 16-bit | 2.5x |
| KG entity cache | 16-bit | 3.0x |
| Reasoning fragments | 12-bit | 4.0x |
| Entropy history | 8-bit | 6.0x |

```bash
python train_compressed.py
```

### Simple Training

```bash
python train_simple.py
```

### Tokenizer Training

```bash
python train_tokenizer.py \
    --corpus datasets/styles.jsonl \
    --output-dir ./tokenizer \
    --vocab-size 50000 \
    --max-context 10000
```

### Training Data

- **File:** `datasets/styles.jsonl`
- **Size:** 7,843 conversation samples, approximately 26KB
- **Format:** `{"provider": "chatgpt", "style_label": "qa", "user_input": "...", "assistant_reply": "...", "turn_id": "..."}`
- **Split:** 80/20 train/val (6,274 training, 1,569 validation)
- **Sources:** Exported conversation samples from ChatGPT, Claude, and Gemini

---

## RLHF Suite

The `RLHF/` directory contains three production-grade modules:

### `RLHF/rlhf.py` — Alignment Pipeline

Full RLHF training implementation:
- PPO trainer with clipping, value function, and KL penalty
- DPO (Direct Preference Optimization) trainer
- Reward model training scaffold

### `RLHF/inference_optimizations.py` — Serving Optimizations

| Class | Description |
|---|---|
| `OptimizedAttention` | Automatic FA2/SDPA kernel selection |
| `PagedKVCache` | Paged attention KV-cache management |
| `SpeculativeDecoder` | Speculative decoding with draft model |
| `BestOfNSampler` | Best-of-N sampling with reward scoring |
| `MCTSGenerator` | Monte Carlo Tree Search generation |
| `compile_model` | `torch.compile` wrapper with backend selection |

### `RLHF/model_merging.py` — Model Fusion

| Class / Function | Description |
|---|---|
| `ModelMerger` | Task Arithmetic, TIES, SLERP, DARE merging strategies |
| `ModelSoup` | Uniform and weighted model soup averaging |
| `EnsemblePolicy` | Ensemble decoding across multiple model checkpoints |
| `layer_wise_interpolation` | Per-layer interpolation between two checkpoints |

---

## Export and Deployment

### Native PyTorch Serving (Recommended)

Native `.pt` checkpoints preserve all advanced modules. This is the only deployment path that retains KG attention, neural memory, continual learning, uncertainty quantification, and the reasoning engine.

```python
import torch
from aeron import NeuralNetConfig, TransformerNeuralNetBackbone

checkpoint = torch.load("checkpoints/lonpt/lonpt_syntactic_002700.pt")
config = NeuralNetConfig(...)  # Match training config
model = TransformerNeuralNetBackbone(config)
model.load_state_dict(checkpoint['model'])
model.eval()

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    decoder_input_ids=decoder_input_ids,
    use_cache=False
)
```

### GGUF Export (Stripped / Benchmarking Only)

```bash
python pt-gguf.py --checkpoint checkpoints/lonpt/lonpt_syntactic_002700.pt --output aeron.gguf
```

**Warning:** GGUF export strips all advanced modules. Of the full checkpoint's tensors, only the basic transformer blocks are exported. The resulting model is suitable only for basic text generation benchmarking and does not represent Aeron's research capabilities.

Specifically lost in GGUF export:
- KnowledgeGraphAttention
- NeuralMemoryNetwork
- ContinualLearningModule
- UncertaintyQuantification
- ActiveLearningManager
- AeronReasoningEngine (ToT + Scratchpad)
- Multimodal fusion path

### ONNX Export (Visualization Only)

ONNX export via `export_model_to_onnx` produces a logits-only graph intended for Netron architecture visualization. It is not suitable for inference.

### TorchScript Export

```python
scripted = torch.jit.script(model)
# or
traced = torch.jit.trace(model, example_inputs)
# Deploy via LibTorch (C++)
```

TorchScript preserves custom modules if properly annotated with type hints.

### Deployment Config Generation

```bash
python generate_lonpt_config.py   # Generate config.json from checkpoint
python validate_json_files.py      # Validate generated config files
python save_lonpt_tokenizer_files.py  # Extract tokenizer for HuggingFace-style deployment
python setup_ollama.py             # Automated GGUF conversion + Ollama registration
```

---

## Consumer Hardware Scaling

The default 3.85B configuration requires approximately 20-30GB VRAM for training. For consumer hardware, use reduced configurations:

```python
from aeron import NeuralNetConfig, TransformerNeuralNetBackbone

# Small config (~100-150M params, 4-6GB VRAM)
config = NeuralNetConfig(
    d_model=512,
    nhead=8,
    num_kv_heads=2,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    max_position_embeddings=2048,
    num_tot_branches=2,
    num_scratchpad_slots=16
)

# Medium config (~400-600M params, 8-12GB VRAM)
config = NeuralNetConfig(
    d_model=1024,
    nhead=16,
    num_kv_heads=4,
    num_encoder_layers=12,
    num_decoder_layers=12,
    dim_feedforward=4096,
    max_position_embeddings=4096,
    num_tot_branches=4,
    num_scratchpad_slots=32
)

# Large config (~1.5B params, 16-20GB VRAM)
config = NeuralNetConfig(
    d_model=1536,
    nhead=24,
    num_kv_heads=8,
    num_encoder_layers=24,
    num_decoder_layers=24,
    dim_feedforward=6144,
    max_position_embeddings=8192,
    num_tot_branches=4,
    num_scratchpad_slots=64
)

model = TransformerNeuralNetBackbone(config)
```

For all reduced configs, apply compressed training to further reduce memory footprint:

```bash
python train_compressed.py
```

---

## Repository Structure

```
aeron/
├── aeron.py                          # Core model (3.85B, 4600+ lines, DO NOT EDIT)
├── tokenizer_mux.py                  # EnhancedBPETokenizer (to be simplified)
├── requirements.txt
│
├── RLHF/
│   ├── rlhf.py                       # PPO, DPO, reward modeling
│   ├── inference_optimizations.py    # FA2/SDPA, PagedKV, speculative decoding
│   └── model_merging.py              # Task Arithmetic, TIES, SLERP, DARE
│
├── lonpt/                            # LONPT training pipeline
│   ├── train_lonpt.py
│   ├── lonpt_graph_transformer.py
│   ├── lonpt_hardware_profiler.py
│   ├── lonpt_hpf_core.py
│   ├── lonpt_act_transformer.py
│   ├── lonpt_akap_sequencer.py
│   ├── lonpt_pncec_controller.py
│   ├── lonpt_integration_controller.py
│   ├── lonpt_core.py
│   └── aeron_adapter.py
│
├── training_methods/
│   ├── compressed_trainer.py         # CompressedMultiModalTrainer
│   └── COMPRESSED_TRAINING.md
│
├── elryse/                           # Experimental (NOT integrated with Aeron)
│   ├── sacred_fbs_tokenizer.py
│   ├── harmonic_breath_field_fbs_enhanced.py
│   └── test_sacred_fbs.py
│
├── datasets/
│   └── styles.jsonl                  # 7,843 conversation samples
│
├── checkpoints/                      # Training checkpoints (.pt)
├── visualizations/                   # Visualization outputs
├── visualizations_sota/              # SOTA++ architecture visualizations
│
├── train_lonpt.py                    # LONPT training entry point
├── train_compressed.py               # Compressed training entry point
├── train_simple.py                   # Basic training entry point
├── train_tokenizer.py                # BPE tokenizer training
├── trainer.py                        # Reference trainer
├── pt-gguf.py                        # GGUF conversion (strips advanced modules)
├── visualize_aeron.py                # Architecture/training visualization suite
├── inspect_checkpoint.py             # Checkpoint inspection
├── deep_checkpoint_analysis.py       # Parameter distributions, layer stats
├── compare_checkpoints.py            # Compare two checkpoints
├── load_lonpt_model.py               # Load and test LONPT checkpoint
│
├── MODELCARD.md                      # Technical model card (v4.0.1)
├── AGENTS.md                         # Developer and agent guidance
└── LONPT_TUI_GUIDE.md                # TUI training guide
```

---

## Built Using

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Visualization
- [NetworkX](https://networkx.org/) - Graph analysis for topology visualization
- [Plotly](https://plotly.com/) - Interactive 3D visualization

---

## Authors

- **treyr** - *Primary architect and developer*

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Tokenizer Canonicalization (SOTA++ Hardening)

Tokenizer implementation now follows a strict single-source policy:

- Canonical implementation: `tokenizer/tokenizer_mux.py`
- Backward-compatible shim: `tokenizer_mux.py`
- Package exports: `tokenizer/__init__.py`

This removes dual-file drift while preserving existing imports used by training and inference scripts.

### Startup Validation

Run strict tokenizer startup contract validation:

```bash
python scripts/tokenizer_startup_validate.py
```

This writes:

- `reports/tokenizer/tokenizer_startup_validation.json`

### Non-Pytest Quality Suite

Run tokenizer hardening checks and generate machine + human artifacts:

```bash
python scripts/tokenizer_quality_runner.py
```

This writes:

- `reports/tokenizer/tokenizer_quality_report.md`
- `reports/tokenizer/tokenizer_quality_manifest.json`

The quality suite validates:

- startup asset hashing and contract checks
- single authoritative caching behavior
- structured payload guardrails (depth/size)
- async timeout enforcement
- per-instance circuit breaker isolation
- fail-loud image lane behavior when required adapter is missing
- root shim identity with canonical module
- multimodal text + structured success path

---

## Session Closeout and Distill Flow

<div align="center">

[![Tokenizer Canonical](https://img.shields.io/badge/tokenizer-canonical%20module%20active-0A7B83.svg)]()
[![Runtime Contract](https://img.shields.io/badge/runtime-fail--loud-critical.svg)]()
[![Startup Validation](https://img.shields.io/badge/startup%20validation-pass-success.svg)]()
[![Quality Suite](https://img.shields.io/badge/quality%20suite-8%2F8%20pass-success.svg)]()
[![Compatibility](https://img.shields.io/badge/import%20compat-root%20shim%20enabled-2D6A4F.svg)]()

</div>

This repository session is closed with `aeron.py` treated as stable for handoff. The tokenizer stack is now canonicalized and hardened for clean-repo migration.

### Canonical Tokenizer Layout

- Canonical implementation: `tokenizer/tokenizer_mux.py`
- Backward-compatible import shim: `tokenizer_mux.py`
- Package export surface: `tokenizer/__init__.py`

### Distill-The-Flow (VPS)

Use this sequence on the clean VPS repository after selecting files:

```bash
python -m py_compile tokenizer/tokenizer_mux.py tokenizer_mux.py \
  scripts/tokenizer_startup_validate.py scripts/tokenizer_quality_runner.py

python scripts/tokenizer_startup_validate.py
python scripts/tokenizer_quality_runner.py
```

### Required Validation Outputs

After the commands above, verify these artifacts exist:

- `reports/tokenizer/tokenizer_startup_validation.json`
- `reports/tokenizer/tokenizer_quality_report.md`
- `reports/tokenizer/tokenizer_quality_manifest.json`

### Suggested Minimal Carryover Set

For a clean Aeron baseline, copy at least:

- `aeron.py`
- `tokenizer/tokenizer_mux.py`
- `tokenizer_mux.py`
- `tokenizer/__init__.py`
- `tokenizer/vocab.json`
- `tokenizer/merges.txt`
- `scripts/tokenizer_startup_validate.py`
- `scripts/tokenizer_quality_runner.py`
- `MODELCARD.md`
- `README.md`

### Release Note for This Session

- Tokenizer runtime is now fail-loud for missing required image adapter/model on image requests.
- Async preprocessing timeout is enforced with `asyncio.wait_for`.
- Circuit breaker state is per-instance and modality-scoped.
- Structured payloads are bounded by depth and serialized length guardrails.
- Root tokenizer module no longer carries implementation drift risk.
