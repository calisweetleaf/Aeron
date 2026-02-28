# Aeron Untrained Model Analysis Report

Generated: 2026-02-27T00:54:51.469454

## Model Configuration

| Parameter | Value |
|-----------|-------|
| vocab_size | 10,000 |
| d_model | 256 |
| nhead | 8 |
| num_encoder_layers | 4 |
| num_decoder_layers | 4 |
| dim_feedforward | 1024 |
| dropout | 0.1 |
| max_position_embeddings | 256 |
| num_multimodal_layers | 4 |
| vision_d_model | 384 |

## Model Statistics

- **Total Parameters**: 49,822,468 (49.82M)
- **Trainable Parameters**: 49,822,468 (49.82M)
- **Total Modules**: 498

## Advanced Components

### Neural Memory Network
- Memory Size: 1000 slots
- Memory Dimension: 512
- Number of Heads: 8

### Uncertainty Quantification
- MC Samples: 10
- Ensemble Members: 5
- Calibration Bins: 15

### Knowledge Graph Attention
- KG Dimension: 128
- Number of Relations: 50
- Max Entities: 100,000

### Continual Learning
- EWC Lambda: 1000.0
- Max Tasks: 100
- Adaptive Lambda: 1000.00

## Generated Visualizations

### Architecture
- `architecture_overview.png`: High-level component diagram
- `encoder_decoder_structure.png`: Detailed layer structure
- `model_topology.png`: Network graph of all modules

### Initialization Analysis
- `initialization_patterns.png`: Weight distributions by component
- `layer_wise_statistics.png`: Statistics across layers

### Component Visualizations
- `neural_memory_network.png`: Memory network structure
- `uncertainty_quantification.png`: UQ module details
- `knowledge_graph_attention.png`: KG attention structure
- `continual_learning_module.png`: CL module details
- `multimodal_fusion.png`: Multi-modal fusion architecture
- `attention_head_structure.png`: Attention head analysis

## Notes

This analysis was performed on an **untrained** model. The visualizations show:
- Weight initialization patterns (Xavier/Gaussian)
- Architecture topology and component relationships
- Structural properties before training

For trained model analysis, use the training-aware visualization scripts.
