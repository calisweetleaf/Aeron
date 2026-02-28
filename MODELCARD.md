# Model Card: Aeron TransformerNeuralNetBackbone

## 1. Model Identity

- **Model Name:** Aeron TransformerNeuralNetBackbone
- **Repository:** `aeron`
- **Primary Implementation:** `aeron.py`
- **Tokenizer Runtime Import:** `tokenizer_mux.py`
- **Tokenizer Asset Directory:** `tokenizer/`
- **Model Card Version:** 4.0.1 (Structural Corrections + RLHF Suite)
- **Model Card Last Updated:** February 27, 2026
- **Authors / Maintainers:** treyr (project maintainer)
- **License:** MIT

## 2. Executive Summary

Aeron is a production-oriented, research-forward transformer backbone that combines a configurable encoder-decoder architecture with optional advanced subsystems for:

- multimodal fusion (text + image)
- knowledge graph-enhanced attention
- neural memory augmentation
- continual learning hooks
- uncertainty quantification
- active learning sample selection

The codebase is architecturally rich and implementation-heavy, with practical concerns addressed (ONNX export path, fallbacks, caching, runtime toggles, error isolation around advanced modules). At the same time, **current readiness is pre-production for claims-heavy deployment** because end-to-end benchmark/evaluation evidence is not yet documented in this repository for the present configuration.

## 3. Intended Use and Scope

### 3.1 Intended Uses

Aeron is suitable for:

- experimentation with composite transformer systems
- architecture research for multimodal and uncertainty-aware NLP/ML systems
- building a foundation for custom task fine-tuning
- prototype-to-deployment workflows that require ONNX export

### 3.2 Out-of-Scope / Not Yet Supported Claims

At this time, do not claim that Aeron is:

- a fully benchmarked SOTA model on public leaderboards
- a safety-certified model for high-stakes autonomous operation
- a validated medical, legal, or financial decision engine
- a guaranteed low-latency edge model

## 4. Current Development and Readiness Status

### 4.1 Implementation Status

The following are implemented in code:

- configurable transformer encoder-decoder backbone
- multimodal path via patch embedding and fusion layers
- knowledge graph attention module with caching/statistics
- neural memory network module
- continual learning module
- uncertainty quantification module (MC-style, ensemble-style, evidential path)
- active learning manager
- streaming JSONL dataset loading pipeline
- ONNX export with export-mode toggling

### 4.2 Training / Evaluation Status (Important)

- This repository currently documents architecture and training scaffolding well, but does **not** include a finalized, reproducible benchmark report for the exact current stack and tokenizer settings.
- The tokenizer path is wired and operational, but tokenizer assets in `tokenizer/` indicate an early-stage state (details in Section 8).
- Therefore, model quality claims should be treated as **unverified until run-specific eval reports are attached**.

## 5. Architecture Details

## 5.1 Core Backbone (`aeron.py`)

The `NeuralNetConfig` default architecture parameters are:

- `vocab_size = 50000`
- `d_model = 2048`
- `nhead = 32`
- `num_kv_heads = 8`
- `num_encoder_layers = 32`
- `num_decoder_layers = 32`
- `dim_feedforward = 8192`
- `dropout = 0.0`
- `activation = "silu"`
- `rms_norm_eps = 1e-6`
- `layer_norm_eps = 1e-5`
- `rope_theta = 500000.0`
- `max_position_embeddings = 32768`
- `num_multimodal_layers = 6`
- `patch_size = 16`
- `num_channels = 3`
- `vision_d_model = 768`
- `num_tot_branches = 4`
- `num_scratchpad_slots = 64`
- `max_reasoning_steps = 3`
- `reasoning_complexity_threshold = 0.3`

Core primitives include:

- multi-head attention with optional relative positional bias
- feedforward transformer blocks
- positional encoding module
- encoder and decoder stacks
- output projection to vocabulary logits

## 5.2 Optional Advanced Subsystems

Aeron instantiates the following enhanced modules in the backbone:

- `KnowledgeGraphAttention`
- `NeuralMemoryNetwork`
- `ContinualLearningModule`
- `UncertaintyQuantification`
- `ActiveLearningManager`

### 5.2.1 Runtime Gating Behavior

In `TransformerNeuralNetBackbone.forward`, advanced modules are applied when `export_mode == False`. Each subsystem is wrapped with independent error handling; failures fall back to non-enhanced behavior while preserving a forward result payload.

### 5.2.2 Export Mode

`set_export_mode(True)` toggles export-oriented behavior and moves model to eval mode. This is used by `export_model_to_onnx(...)` to produce a cleaner ONNX graph path.

## 5.3 Multimodal Path

If `vision_inputs` are supplied:

- image inputs are patch-embedded
- text embeddings and vision embeddings are fused through configured multimodal fusion layers
- resulting fused representations feed the encoder path

If no vision input is provided, the text-only path is used.

## 6. Input and Output Contract

### 6.1 Primary Inputs

Expected tensors include:

- `input_ids`: shape `(batch, seq_len)`
- `attention_mask`: optional; if omitted, auto-filled as all-ones
- `vision_inputs`: optional image tensor
- `decoder_input_ids`: optional target-side IDs
- `decoder_attention_mask`: optional causal/target mask

### 6.2 Output Dictionary

Backbone forward returns a structured dictionary. The complete set of keys as of v4.0.1:

| Key | Type | Description |
|---|---|---|
| `logits` | `Tensor(batch, seq, vocab_size)` | Primary output |
| `encoder_attentions` | `Tensor or None` | Encoder self-attention weights |
| `decoder_self_attentions` | `Tensor or None` | Decoder self-attention weights |
| `decoder_cross_attentions` | `Tensor or None` | Decoder cross-attention weights |
| `knowledge_graph_enhanced` | `bool` | True if KG attention was applied |
| `neural_memory_enhanced` | `bool` | True if episodic memory was applied |
| `memory_statistics` | `dict` | Memory access patterns and slot utilization |
| `uncertainty_estimates` | `dict` | MC/Ensemble/Evidential uncertainty outputs |
| `tot_branch_scores` | `Tensor(batch, num_tot_branches) or None` | Per-branch quality scores from ToT |
| `scratchpad_stats` | `dict` | `{used_slots, total_slots, step_counter}` |
| `reasoning_info` | `dict` | Full reasoning diagnostics: complexity, branch_weights, tot_resolved_contradictions, steps taken |

Downstream consumers should pin to a strict subset of these fields. The research-diagnostic keys (reasoning_info, scratchpad_stats) are not guaranteed stable across minor versions.

This output contract is useful for diagnostics and research introspection, but downstream consumers must define a strict schema for production APIs.

## 7. Training Data Pipeline

Aeron includes two data paths:

- small in-memory sample dataset path for demo/testing
- streaming JSONL dataset path for larger training sets (`JsonlTextDataset`)

`prepare_training_dataloader(...)` scans `datasets/` in configurable order (default via env):

- `styles`
- `reports`
- `code`

The JSONL path includes formatter functions for each dataset type and builds tokenized batches with attention masks derived from pad token IDs.

## 8. Tokenizer Card (Current State)

## 8.1 Active Tokenizer Components

Aeron tokenizer integration uses:

- `TokenizerConfig`
- `EnhancedBPETokenizer`
- `AeronTokenizer` adapter wrapper in `aeron.py`

Runtime defaults point to:

- vocab: `./tokenizer/vocab.json`
- merges: `./tokenizer/merges.txt`

## 8.2 Observed Tokenizer Asset State (Repository Snapshot)

From current repository files:

- `tokenizer/vocab.json` shows a compact vocabulary with byte tokens and special tokens
- `tokenizer/merges.txt` contains only a version header (`#version: 0.2`) and no learned merge rules
- `tokenizer/tokenizer_config.json` reports `vocab_size = 268`

### 8.2.1 Practical Interpretation

This tokenizer is currently in an early-stage/bootstrapped configuration, not a mature learned BPE merge-space. It is operational and consistent for development, but not yet the kind of tokenizer expected for high-capacity language modeling quality.

That means current model behavior can be architecturally validated, but quality claims tied to subword efficiency or linguistic compression should be deferred until tokenizer training is completed.

## 9. Safety, Reliability, and Risk

## 9.1 Known Strengths

- robust modular decomposition for experimentation
- explicit fallback paths around optional advanced modules
- export compatibility path for deployment pipeline integration
- support for uncertainty and active-learning introspection

## 9.2 Known Risks and Limitations

- no consolidated benchmark suite/report in repo for current stack
- tokenizer merge training appears incomplete
- advanced module presence can create false confidence if eval evidence is missing
- output structure is broad; production API consumers must pin strict fields
- additional code paths (demo/experimental utilities) should be separated from hardened serving runtime

## 9.3 High-Stakes Use Guidance

For any high-stakes deployment:

- require full task-specific validation
- run robustness and failure-mode testing
- include human review and rollback controls
- monitor uncertainty outputs as heuristics, not guarantees

## 10. Bias, Fairness, and Responsible Use

Because data provenance and balancing are not yet fully documented in this repository snapshot, fairness properties cannot be asserted. Before external deployment:

- document dataset sources and filtering
- evaluate demographic and domain skew
- run subgroup performance audits
- publish known limitations transparently

## 11. Evaluation Evidence Status

## 11.1 What Exists

- architecture implementation
- training/eval scaffolding
- complexity/benchmark helper utilities
- ONNX export path

## 11.2 What Is Missing for Production Claims

- pinned benchmark matrix with dataset versions and seeds
- reproducible quality metrics (e.g., perplexity/accuracy/F1/task KPIs)
- latency/memory results for target deployment hardware and batch regimes
- failure case taxonomy and mitigation outcomes

## 11.3 Recommended Evaluation Bundle (Before Public Claiming)

Minimum publishable evidence:

- training config and seed manifest
- tokenizer build manifest (vocab/merges generation recipe)
- held-out metrics with confidence intervals
- ablations: base vs +KG vs +memory vs +continual vs +uncertainty
- serving latency and memory at representative sequence lengths

## 12. Reproducibility Notes

- Primary model implementation: `aeron.py`
- Tokenizer module appears in both `tokenizer_mux.py` and `tokenizer/tokenizer_mux.py` in repository snapshot
- `aeron.py` currently imports from root module path (`from tokenizer_mux import ...`)
- Tokenizer assets resolve under `tokenizer/` by default
- Dataset directory default: `datasets/`
- Export artifact default path used in demo flow: `exports/transformer_backbone.onnx`

To reduce drift risk, keep tokenizer module source-of-truth singular and version-pin tokenizer assets with checksums.

## 13. Deployment Guidance

## 13.1 Recommended Deployment Profile

- enable `export_mode` for serving/export pipelines
- separate training-only logic from inference runtime
- pin model and tokenizer artifacts by immutable version tags
- define strict output schema in serving layer

## 13.2 Monitoring Checklist

- request throughput and latency by sequence length bucket
- memory utilization and OOM incidents
- error rates by subsystem
- uncertainty distribution drift over time
- tokenization anomalies and unknown-token rates

## 14. Model Governance

## 14.1 Versioning Policy (Recommended)

For each release, publish:

- model commit hash
- tokenizer commit hash and asset hashes
- config diff vs previous release
- metric diff vs previous release

## 14.2 Change Control (Recommended)

Any change to tokenizer vocab/merges, model architecture defaults, or data formatter behavior should trigger:

- retraining or revalidation
- benchmark refresh
- model card update

## 15. Practical Summary

Aeron is a serious architecture foundation with strong extensibility and real engineering depth. The right way to present it today is:

- **implementation maturity:** high for architecture scaffolding
- **claim maturity:** moderate pending consolidated eval proof
- **tokenizer maturity:** functional but currently early-stage (merge training incomplete)

With tokenizer training completion and disciplined benchmark publishing, this stack can present as a full lab-grade release with defensible claims.

---

## 16. SOTA++ Architecture Upgrade (February 2026)

### 16.1 Upgrade Overview

Aeron's backbone underwent a comprehensive architectural upgrade integrating state-of-the-art transformer techniques used by leading production models (LLaMA 3, Mistral, DeepSeek-V2, Qwen-2, Gemma 2). This upgrade transforms the backbone from a standard post-norm encoder-decoder into a SOTA++ pre-norm architecture with advanced attention, feed-forward, normalization, and reasoning subsystems.

**Upgrade Date:** February 27, 2026
**Affected File:** `aeron.py`
**Backward Compatibility:** Checkpoint format differs from pre-upgrade; existing checkpoints require migration.
**Visualization Outputs:** `visualizations_sota/` (new subdirectory, separate from legacy `visualizations_untrained/`)

### 16.2 Architectural Changes Summary

| Component | Pre-Upgrade | Post-Upgrade (SOTA++) | Rationale |
|---|---|---|---|
| Normalization | `nn.LayerNorm` (post-norm) | `AeronRMSNorm` (pre-norm) | 10-15% faster, better deep gradient flow |
| Position Encoding | Learned/Sinusoidal additive | **RoPE-only for text**; absolute sinusoidal PE retained for multimodal fusion path only | Relative position awareness, length extrapolation; absolute PE was incorrectly applied to text path in v4.0.0, corrected in v4.0.1 |
| Attention | Standard Multi-Head Attention | Grouped Query Attention (GQA) + RoPE | 2-4x KV-cache reduction, FlashAttention-2 compatible |
| Feed-Forward | Linear-GELU-Linear | SwiGLU (gate-up-down) | Consistently outperforms GELU/ReLU FFN |
| Architecture Pattern | Post-norm (norm after sublayer) | Pre-norm (norm before sublayer) | Superior gradient flow for deep networks (24+ layers) |
| Embedding Tying | Separate input/output embeddings | Weight-tied (shared) | Saves ~vocab_size x d_model params, improves quality |
| Output Projection | `nn.Linear(d_model, vocab_size)` w/ bias | `nn.Linear(d_model, vocab_size, bias=False)` | Consistent with tied weights, reduced param count |
| Final Norm | Single shared `final_layer_norm` | Separate `encoder_norm` + `decoder_norm` (RMSNorm) | Eliminates double-norm bug |
| Reasoning | None | AeronReasoningEngine (ToT + Scratchpad) | Neural-level multi-hypothesis reasoning |

### 16.3 New Core Primitives

#### 16.3.1 AeronRMSNorm

Root Mean Square Layer Normalization. Replaces all `nn.LayerNorm` instances throughout the model.

```
Output = (x / sqrt(mean(x^2) + eps)) * weight
```

- Skips mean-centering step (vs LayerNorm which computes both mean and variance)
- ~10-15% wallclock speedup on GPU
- Configured via `rms_norm_eps` (default: `1e-6`)
- Used by: LLaMA, Mistral, Gemma, DeepSeek, Qwen

#### 16.3.2 AeronRotaryEmbedding (RoPE)

Rotary Position Embeddings encode relative position information by rotating Q/K vectors in attention.

- **Mechanism:** Applies rotation matrix to query and key vectors based on position
- **Cache:** Pre-computes cos/sin values up to `max_position_embeddings`, dynamically extends
- **Extrapolation:** Naturally handles sequences longer than training length
- **Scaling:** Optional `rope_scaling` factor for context window extension
- **Applied to:** Self-attention only (NOT cross-attention, which attends across sequences)
- **Config:** `rope_theta=500000.0` (default v4; extended context), `rope_scaling=None`

#### 16.3.3 MultiHeadAttention (GQA)

Grouped Query Attention shares K/V projection heads across groups of query heads, reducing memory consumption and enabling efficient KV-caching during generation.

- **Config:** `nhead=32` (query heads), `num_kv_heads=8` (KV heads), `num_kv_groups=4` (default v4)
- **Special cases:** `num_kv_heads == nhead` → standard MHA; `num_kv_heads == 1` → Multi-Query Attention (MQA)
- **Backend:** Uses `torch.nn.functional.scaled_dot_product_attention` for automatic FlashAttention-2/memory-efficient kernel selection
- **Fallback:** Manual attention computation if SDPA unavailable
- **Bias-free:** All projection matrices (`q_proj`, `k_proj`, `v_proj`, `out_proj`) use `bias=False`

#### 16.3.4 FeedForwardNetwork (SwiGLU)

SwiGLU gating mechanism replaces the standard 2-layer FFN.

```
Output = Dropout(W_down @ (SiLU(W_gate @ x) * (W_up @ x)))
```

- **Three matrices** instead of two (gate, up, down)
- **Intermediate size:** `2/3 * dim_feedforward`, rounded to nearest 256 boundary for hardware efficiency
- **Activation:** SiLU (Sigmoid Linear Unit) for the gating path
- **Bias-free:** All three projection matrices use `bias=False`

#### 16.3.5 TransformerEncoderLayer / TransformerDecoderLayer (Pre-Norm)

Pre-norm architecture applies normalization BEFORE each sublayer, with residual connections wrapping the normalized path.

**Encoder Flow:**
```
residual = x
x = RMSNorm(x) → Self-Attn(GQA+RoPE) → x + residual
residual = x
x = RMSNorm(x) → SwiGLU FFN → x + residual
```

**Decoder Flow:**
```
residual = x
x = RMSNorm(x) → Masked Self-Attn(GQA+RoPE) → x + residual
residual = x
x = RMSNorm(x) → Cross-Attn(GQA, no RoPE) → x + residual
residual = x
x = RMSNorm(x) → SwiGLU FFN → x + residual
```

### 16.4 Reasoning Engine Integration

A neural-level reasoning system is integrated directly into the forward pass, positioned between the enhancement pipeline (KG/Memory/CL) and the decoder. This provides o1-class "thinking before speaking" capability.

#### 16.4.1 AeronScratchpadMemory

Neural working memory. In v4, this role is fulfilled by `AeronInternalScratchpad` with 64 differentiable memory slots (see Section 17.3.3 for the v4 implementation).

| Capability | Description |
|---|---|
| **Read** | Importance-weighted attention retrieval over memory slots |
| **Write** | Gated update mechanism (lerp between old and new content) |
| **Consolidation** | Cosine-similarity based merging of related slots |
| **Contradiction Detection** | Pairwise contradiction scoring between memory slots |
| **Eviction** | Learned importance scorer determines which slots to overwrite |

#### 16.4.2 AeronTreeOfThoughtLayer

Neural branch exploration within the transformer hidden state.

1. **Branch Generation:** 4 parallel hypothesis generators produce candidate reasoning paths
2. **Branch Evaluation:** Learned critic network scores each branch against the original representation
3. **Pruning:** Top-K (K=2) branches survive based on critic scores
4. **Refinement:** Surviving branches pass through a shared self-attention + FFN refinement step
5. **Merging:** Quality-weighted combination of refined branches
6. **Confidence Gating:** Learned gate controls blend ratio between tree reasoning output and original representation

#### 16.4.3 AeronReasoningEngine (Orchestrator)

**Note:** This section describes the SOTA++ intermediate design. The final v4 implementation is documented in Section 17.3.5. Key differences: v4 uses a strict ToT-first mandate (no CoT-style iterative loop), scratchpad write occurs strictly after ToT, and convergence-based early stopping was replaced by a single ToT pass per forward call.

Master controller that orchestrates the scratchpad and tree of thought.

| Feature | Description |
|---|---|
| **Adaptive Halting** | Complexity estimator (threshold=0.3) skips reasoning for simple inputs, saving compute |
| **ToT-First Mandate** | Tree of Thought runs before any scratchpad writes (strict ordering) |
| **Cross-Memory Bridge** | Connects NeuralMemoryNetwork output to ToT branch exploration |
| **Scratchpad Synthesis** | After ToT: write best_branch and surviving branches, then synthesize |
| **Skip Mask** | Simple samples use residual path; complex samples use synthesized reasoning output |

### 16.5 Forward Pass Architecture (Post-Upgrade)

The complete forward pass flow after the SOTA++ upgrade:

```
INPUT: input_ids (batch, seq_len)
  |
  +--> Token Embeddings (50k vocab, d_model=2048)
  |    [NO absolute PE -- RoPE handles position inside each attention layer]
  |
  +--> [Optional] Vision Patch Embedding --> Absolute PE --> Multimodal Fusion
  |
  v
ENCODER STACK (32x Pre-Norm layers)
  |  Each layer: AeronRMSNorm --> GQA Self-Attn (32Q/8KV + RoPE) --> Residual
  |              AeronRMSNorm --> SwiGLU FFN (2048-->5461-->2048) --> Residual
  |
  +--> encoder_norm (AeronRMSNorm)
  |
  v
ENHANCEMENT PIPELINE (sequential, error-isolated):
  1. KnowledgeGraphAttention    -- structured knowledge injection
  2. NeuralMemoryNetwork        -- 1000-slot episodic memory (memory_dim=512)
  3. ContinualLearningModule    -- EWC consolidation
  4. UncertaintyQuantification  -- evidential deep learning heads
  5. ActiveLearningManager      -- training-only, BADGE sampling
  |
  v
REASONING ENGINE (AeronReasoningEngine):
  +--> complexity_gate --> skip entirely if complexity < 0.3
  +--> [TREE OF THOUGHT] 4 branches --> cross-branch attention (O(N^2) on branch summaries)
  |    --> confidence/evidence scoring --> pruning_decider --> contradiction resolution
  |    --> quality-weighted merge
  |    (reads NeuralMemoryNetwork via memory_bridge; KG already in encoder_output;
  |     CL task_embedding conditions branch exploration)
  +--> [WRITE TO SCRATCHPAD] strictly after ToT returns
  |    best_branch --> WORKING slots; survivors --> SHORT_TERM slots
  +--> AeronScratchpadAttention synthesizes across written slots
  +--> Final reasoning projection with residual
  |
  v
DECODER STACK (32x Pre-Norm layers)
  |  Each layer: AeronRMSNorm --> Masked GQA Self-Attn (32Q/8KV + RoPE) --> Residual
  |              AeronRMSNorm --> Cross-Attn (GQA, no RoPE) --> Residual
  |              AeronRMSNorm --> SwiGLU FFN --> Residual
  |
  +--> decoder_norm (AeronRMSNorm)
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

### 16.6 Bug Fixes Addressed

#### 16.6.1 Decoder Ordering Bug (Critical)

**Before:** The decoder ran BEFORE the enhancement pipeline (KG/Memory/CL), meaning it attended to raw, unenhanced encoder output. The KG, memory, and CL modules operated on the encoder output but their results were never seen by the decoder.

**After:** The enhancement pipeline runs first, followed by the reasoning engine, followed by the decoder. The decoder now attends to fully enhanced and reasoning-processed representations.

**Impact:** Substantial quality improvement expected once trained, as the decoder can now leverage all neural memory, knowledge graph, and reasoning capabilities.

#### 16.6.2 Double LayerNorm Bug

**Before:** `encode()` applied `final_layer_norm` to encoder output. Then in `forward()`, the encoder-only code path applied `self.final_layer_norm(enhanced_encoder_output)` again, resulting in double normalization.

**After:** Separate `encoder_norm` and `decoder_norm` (both `AeronRMSNorm`) instances. Each is applied exactly once in its respective method. Forward pass projects directly without re-normalizing.

### 16.7 Updated NeuralNetConfig Parameters

The following parameters were added to `NeuralNetConfig`:

| Parameter | Default | Description |
|---|---|---|
| `num_kv_heads` | 8 | Number of KV heads for GQA |
| `activation` | `"silu"` | Activation function (SwiGLU uses SiLU) |
| `rms_norm_eps` | `1e-6` | RMSNorm epsilon for numerical stability |
| `rope_theta` | `10000.0` | RoPE base frequency |
| `rope_scaling` | `None` | Optional scaling factor for context extension |
| `use_gradient_checkpointing` | `False` | Enable gradient checkpointing for memory efficiency |
| `tie_word_embeddings` | `True` | Share input/output embedding weights |
| `head_dim` | `d_model // nhead` | Derived: per-head dimension |
| `num_kv_groups` | `nhead // num_kv_heads` | Derived: GQA group count |

### 16.8 Updated Output Dictionary

The backbone forward method now returns an additional key:

| Key | Type | Description |
|---|---|---|
| `reasoning_info` | `Dict[str, Any]` | Reasoning engine diagnostics: steps taken, complexity score, convergence status, branch scores, contradiction levels |

All previously documented keys remain unchanged.

### 16.9 Parameter Budget Analysis (3B Config)

At the target configuration (`d_model=2560, nhead=20, num_kv_heads=4, 16+16 layers`):

| Component | Approximate Parameters | Notes |
|---|---|---|
| Token Embeddings | ~128M | 50k × 2560 (shared with output projection via tying) |
| Encoder Stack (16 layers) | ~1.28B | GQA Self-Attn (20Q/4KV) + SwiGLU (2560→6826) per layer |
| Decoder Stack (16 layers) | ~1.49B | GQA Self-Attn + GQA Cross-Attn + SwiGLU per layer |
| Output Projection | 0 (tied) | Shared with token embeddings |
| KnowledgeGraphAttention | ~19M | Inherits GQA + KG injector/bias nets |
| NeuralMemoryNetwork | ~15M | 1K-slot memory + consolidation (scaled) |
| ContinualLearningModule | ~8M | Task embeddings + EWC Fisher tracking |
| UncertaintyQuantification | ~18M | MC/Ensemble/Evidential heads |
| ActiveLearningManager | ~6M | BADGE sampling networks |
| ReasoningEngine | ~42M | Scratchpad + ToT + Bridge |
| **Estimated Total** | **~3.0B** | **Meets the ~3B parameter target scale** |

### 16.10 Visualization Outputs (SOTA++)

New visualization outputs are located in `visualizations_sota/` to preserve the pre-upgrade visualizations in `visualizations_untrained/`.

```
visualizations_sota/
├── architecture/
│   ├── architecture_overview.png      -- Full component diagram with SOTA++ labels
│   ├── encoder_decoder_structure.png  -- Pre-norm layer block detail
│   ├── model_topology.png            -- NetworkX graph of module hierarchy
│   └── topology_data.json            -- Machine-readable topology
├── components/
│   ├── attention_head_structure.png   -- GQA head configuration + RoPE
│   ├── continual_learning_module.png  -- EWC Fisher visualization
│   ├── knowledge_graph_attention.png  -- KG injection paths
│   ├── multimodal_fusion.png         -- Vision-text fusion layers
│   ├── neural_memory_network.png     -- Memory slot analysis
│   └── uncertainty_quantification.png -- UQ estimation heads
├── initialization/
│   ├── initialization_patterns.png    -- SwiGLU/RMSNorm weight distributions
│   └── layer_wise_statistics.png      -- Layer norm/weight statistics
├── untrained_model_report.json        -- Machine-readable analysis
└── README.md                          -- Auto-generated report
```

### 16.11 Verified Component Tests

All SOTA++ components pass functional verification (Feb 27, 2026):

```
Config: d_model=256, nhead=8, num_kv_heads=4, head_dim=32, groups=2
RMSNorm:           torch.Size([2, 10, 256])  ✓
RoPE cos/sin:      torch.Size([10, 32])      ✓
GQA MHA:           out=(2,10,256) w=(2,8,10,10)  ✓
SwiGLU FFN:        intermediate=512, out=(2,10,256)  ✓
Encoder Layer:     (2,10,256)  ✓
Decoder Layer:     (2,10,256)  ✓
Scratchpad Read:   (2,10,256)  ✓
Scratchpad Write:  (2,8,256)   ✓
Tree of Thought:   out=(2,10,256), scores=(2,4), gate=0.501  ✓
Reasoning Engine:  out=(2,10,256), steps=2, complexity=0.521  ✓
```

### 16.12 Migration Notes

#### 16.12.1 Checkpoint Compatibility

Pre-upgrade checkpoints **will not load** directly due to:

- `final_layer_norm` renamed to `encoder_norm` / `decoder_norm`
- `output_projection` bias removed (`bias=False`)
- Attention projections restructured for GQA (`q_proj`, `k_proj`, `v_proj`)
- FFN restructured for SwiGLU (`gate_proj`, `up_proj`, `down_proj`)
- New modules: `reasoning_engine`, `AeronRMSNorm` instances

A migration script mapping old key names to new key names is required for checkpoint conversion.

#### 16.12.2 LONPT Adapter Compatibility

The `lonpt/aeron_adapter.py` should be verified against the new architecture. Key changes that may affect the adapter:

- Norm type change (LayerNorm → RMSNorm)
- Attention head structure change (MHA → GQA)
- FFN structure change (2-layer → SwiGLU 3-layer)
- New reasoning engine module

---

## 17. 4B Scale + ToT+Scratchpad Upgrade (February 2026)

### 17.1 Upgrade Overview

Aeron's backbone was scaled to ~3.85B parameters and the reasoning engine was rebuilt from the ground up with a strict **Tree of Thought → Scratchpad** pipeline replacing the previous iterative CoT-style loop.

**Upgrade Date:** February 27, 2026
**Model Card Version:** 4.0.0
**Affected Files:** `aeron.py`, `tokenizer_mux.py`

### 17.2 Scale Changes

| Parameter | v3 Default | v4 Default | Change |
|---|---|---|---|
| `d_model` | 1024 | 2048 | 2× |
| `nhead` | 16 | 32 | 2× |
| `num_encoder_layers` | 12 | 32 | 2.7× |
| `num_decoder_layers` | 12 | 32 | 2.7× |
| `dim_feedforward` | 4096 | 8192 | 2× |
| `dropout` | 0.1 | 0.0 | Disabled at 4B scale |
| `rope_theta` | 10000.0 | 500000.0 | Extended context |
| `max_position_embeddings` | 4096 | 32768 | 8× (32k context) |
| `max_text_length` (tokenizer) | 4096 | 32768 | Matches model context |

**Estimated parameter count (d_model=2048, 32/32 layers):** ~3.85B
- Embedding: 50k × 2048 = 102M (shared with output via tying)
- Encoder stack (32 layers): ~1,376M
- Decoder stack (32 layers): ~2,048M
- Advanced modules (KG, Memory, CL, UQ, AL): ~160M
- Reasoning engine: ~170M

### 17.3 Reasoning Engine Architecture (v4)

The reasoning engine was completely redesigned. **Key mandate: Tree of Thought FIRST → then write to Scratchpad. NO CoT anywhere. NO MoE.**

#### 17.3.1 New Forward Pipeline

```
encoder_output (KG+Memory+CL enhanced)
    ↓
AeronReasoningEngine:
  1. complexity_gate → skip if complexity < 0.3 (saves compute for simple inputs)
  2. [TREE OF THOUGHT] — branches explore, evaluate, prune, resolve contradictions
     - reads neural_memory context via memory_bridge (Memory #1)
     - KG knowledge already baked into encoder_output (Memory #2)
     - CL task embedding conditions exploration (Memory #3)
  3. [WRITE TO SCRATCHPAD] — strictly after ToT returns
     - best_branch written to WORKING slots
     - surviving branches written to SHORT_TERM slots
     - ScratchpadAttention synthesizes across branch slots
  4. synthesized_output → decoder
```

#### 17.3.2 AeronScratchpadAttention

Multi-head attention over 64 differentiable memory slots with metadata enhancement (slot type, priority, timestamp).

- `nhead = max(4, d_model // 64)` — scales with model size
- Slot relevance scoring via learned importance/recency/relevance scorers
- Metadata embeddings: `MemorySlotType` (5 values) × `MemoryPriority` (5 values) × temporal encoding

#### 17.3.3 AeronInternalScratchpad

64 differentiable memory slots with learnable key/value parameters.

| Method | Description |
|---|---|
| `read(query)` | Attend over slot values with metadata-enhanced keys |
| `write(content, slot_type, priority)` | Soft weighted update to slot values |
| `synthesize(tot_output, x)` | Read after write → concat → project back to d_model |
| `get_stats()` | Used/total slots, step counter |

#### 17.3.4 AeronTreeOfThoughtProcessor

Parallel hypothesis exploration with dialectical contradiction resolution.

1. **Branch generation:** `num_tot_branches=4` parallel generators produce hypothesis representations
2. **Cross-branch attention:** `TransformerEncoderLayer` processes all branches concatenated
3. **Quality estimation:** confidence + evidence assessors score each branch
4. **Pruning:** `pruning_decider` per-branch keep/prune decision
5. **Contradiction resolution:** pairwise detector + dialectical `synthesis_resolver` merges contradictory branches
6. **Quality-weighted merge:** softmax-weighted combination of surviving branches

#### 17.3.5 AeronReasoningEngine (v4 Orchestrator)

| Feature | Description |
|---|---|
| **Adaptive halting** | `complexity_gate` (threshold=0.3) skips reasoning for simple inputs |
| **ToT-first mandate** | Tree of Thought runs BEFORE any scratchpad writes |
| **Memory wiring** | Neural memory context → `memory_bridge` → ToT; task embedding → ToT conditioning |
| **Scratchpad synthesis** | After ToT: write best_branch + survivors, then `synthesize()` |
| **Skip mask** | Simple samples use residual; complex samples use synthesized output |

### 17.4 New Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_tot_branches` | 4 | Number of parallel ToT hypothesis branches |
| `num_scratchpad_slots` | 64 | Number of differentiable scratchpad memory slots |
| `max_reasoning_steps` | 3 | Reserved (used as depth hint, ToT runs once per forward) |
| `reasoning_complexity_threshold` | 0.3 | Below this, skip reasoning entirely |

### 17.5 New Output Dictionary Keys

| Key | Type | Description |
|---|---|---|
| `tot_branch_scores` | `Tensor(batch, num_branches)` or `None` | Per-branch quality scores from ToT |
| `scratchpad_stats` | `Dict` | `{used_slots, total_slots, step_counter}` |
| `reasoning_info` | `Dict` | Full reasoning diagnostics including `complexity`, `branch_weights`, `tot_resolved_contradictions` |

### 17.6 Removed in v4

The following old reasoning components were removed:
- `AeronScratchpadMemory` (replaced by `AeronInternalScratchpad` + `AeronScratchpadAttention`)
- `AeronTreeOfThoughtLayer` (replaced by `AeronTreeOfThoughtProcessor`)
- Old `AeronReasoningEngine` iterative loop with CoT-style convergence detection

### 17.7 Forward Pass Signature Change

`TransformerNeuralNetBackbone.forward()` now accepts:
- `task_id: Optional[int] = None` — used to retrieve ContinualLearningModule task embedding for ToT conditioning

### 17.8 Verification

```bash
# Smoke test (< 30 seconds)
python -c "
from aeron import NeuralNetConfig, TransformerNeuralNetBackbone
import torch
config = NeuralNetConfig(
    d_model=256, nhead=4, num_kv_heads=2,
    num_encoder_layers=2, num_decoder_layers=2,
    dim_feedforward=512, vocab_size=50000,
    num_tot_branches=2, num_scratchpad_slots=8,
    max_reasoning_steps=1
)
model = TransformerNeuralNetBackbone(config)
ids = torch.randint(0, 50000, (2, 16))
out = model(input_ids=ids, attention_mask=torch.ones(2,16), decoder_input_ids=ids)
assert out['logits'].shape == (2, 16, 50000)
print('branch_scores:', out['tot_branch_scores'].shape)
print('scratchpad_stats:', out['scratchpad_stats'])
print('ALL CHECKS PASSED')
"
```

### 17.9 v4.0.1 Structural Corrections (February 2026)

Six structural issues were corrected in `aeron.py`:

| # | Issue | Impact |
|---|-------|--------|
| 1 | `use_cache` now raises `NotImplementedError` instead of silently ignoring | Safety |
| 2 | `neural_memory_ctx` now correctly passes `memory_recall` to ToT (with shape guard fallback when `memory_dim ≠ d_model`) | Correctness |
| 3 | Absolute positional encoding removed from text path (RoPE-only); PE applied only to multimodal-fused path | Architecture |
| 4 | SDPA diagnostic `attn_weights` now mask-correct (causal and padding masks applied before softmax) | Correctness |
| 5 | Scratchpad `write()` no longer mutates global state mid-forward; returns differentiable `new_slot_values`; gradient flows through write_gate → synthesis | Training |
| 6 | ToT cross-branch attention reduced from O((N·S)²) to O(N²) via branch-summary pooling | Scalability |

Checkpoint format: unchanged from v4.0.0 (no new parameters added, no parameter shapes changed).

---

## 18. RLHF Suite (February 2026)

### 18.1 Overview

The `RLHF/` directory contains three production-grade modules providing alignment training, inference optimization, and model merging capabilities. These modules operate as wrappers over `TransformerNeuralNetBackbone` and do not modify `aeron.py` directly.

### 18.2 `RLHF/rlhf.py` — Alignment Pipeline

Full reinforcement learning from human feedback implementation:

| Component | Description |
|---|---|
| PPO Trainer | Proximal Policy Optimization with clipping, value function, and KL penalty against reference policy |
| DPO Trainer | Direct Preference Optimization for alignment without explicit reward model |
| Reward Model Training | Scaffold for training a reward model from preference data |

### 18.3 `RLHF/inference_optimizations.py` — Serving Infrastructure

| Class | Description |
|---|---|
| `OptimizedAttention` | Automatic FlashAttention-2 / SDPA kernel selection; falls back to manual attention if unavailable |
| `PagedKVCache` | Paged attention KV-cache management for variable-length batch serving |
| `SpeculativeDecoder` | Speculative decoding with a smaller draft model; accepts or rejects draft tokens against the target model |
| `BestOfNSampler` | Generates N candidates and selects the highest-scoring according to a reward model |
| `MCTSGenerator` | Monte Carlo Tree Search over the token generation graph |
| `compile_model` | `torch.compile` wrapper with configurable backend (inductor, aot_eager, etc.) |

### 18.4 `RLHF/model_merging.py` — Checkpoint Fusion

| Class / Function | Algorithm | Description |
|---|---|---|
| `ModelMerger` | Task Arithmetic | Linear combination of task vectors (fine-tuned minus base) |
| `ModelMerger` | TIES | Trim, Elect Sign, and Merge for conflict resolution across multiple fine-tunes |
| `ModelMerger` | SLERP | Spherical linear interpolation between two checkpoints |
| `ModelMerger` | DARE | Drop and Rescale random parameter delta pruning |
| `ModelSoup` | Uniform / Weighted | Average model weights across a set of checkpoints |
| `EnsemblePolicy` | Ensemble | Decode by averaging logits across multiple loaded model instances |
| `layer_wise_interpolation` | Layer-wise | Per-layer interpolation coefficient between two checkpoints |

### 18.5 Integration Notes

- None of the RLHF modules modify `aeron.py`. They wrap the public `TransformerNeuralNetBackbone` API.
- PPO and DPO trainers require a reference policy (frozen copy of the base model) and a reward model.
- Inference optimizations are independent and can be applied to any forward pass without RLHF training.
- Model merging operates on checkpoint state dicts; no live model instance required for most operations.
- `use_cache=False` must be enforced when using `PagedKVCache` until native KV-cache support is implemented in `aeron.py`.

### 18.6 Readiness

RLHF modules are implemented and structurally present. No end-to-end RLHF training results are documented for the v4.0.1 configuration. Treat as available infrastructure pending benchmark validation.

## Addendum A: Session Closeout (February 27, 2026)

<div align="center">

[![Tokenizer Path](https://img.shields.io/badge/tokenizer-canonical%20path%20set-0A7B83.svg)]()
[![Contract](https://img.shields.io/badge/contract-fail--loud%20runtime-critical.svg)]()
[![Evidence](https://img.shields.io/badge/evidence-validation%20artifacts%20generated-success.svg)]()
[![Compatibility](https://img.shields.io/badge/compatibility-root%20import%20shim%20active-2D6A4F.svg)]()

</div>

### A.1 Canonical Tokenizer Source of Truth

The tokenizer implementation is now treated as canonical in:

- `tokenizer/tokenizer_mux.py`

Backward compatibility is preserved via:

- `tokenizer_mux.py` (re-export shim)

This resolves prior dual-module drift risk while maintaining existing import patterns in training and inference scripts.

### A.2 Runtime Contract Hardening

Tokenizer runtime behavior is now explicitly hardened with fail-loud semantics for contract violations:

- per-instance, modality-scoped circuit breaker execution
- enforced async preprocessing timeout boundaries
- single authoritative tokenization cache path
- structured payload depth and size guardrails
- fail-loud image lane when image tokenizer adapter is required but unavailable
- startup asset validation report with deterministic hash capture

### A.3 Validation Evidence Produced in This Session

The following validation artifacts were generated:

- `reports/tokenizer/tokenizer_startup_validation.json`
- `reports/tokenizer/tokenizer_quality_report.md`
- `reports/tokenizer/tokenizer_quality_manifest.json`

Observed summary from quality manifest:

- suite: `tokenizer_quality_non_pytest`
- total tests: `8`
- passed: `8`
- failed: `0`

### A.4 Claim Discipline for New Clean Repository

For the next repository cut:

- keep `aeron.py` treated as stable unless a new architecture phase explicitly begins
- preserve canonical tokenizer module layout from this addendum
- require startup + quality artifact generation before any production-oriented claims
- keep README and model card statements tied to concrete artifact outputs
