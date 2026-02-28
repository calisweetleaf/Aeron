(.venv) PS C:\Users\trent\aerin> python train_lonpt.py
================================================================================
Aeron + LONPT Training Script
Preset: heavy - Full dataset, 1024 tokens, HPF+PNCEC+AKAP+ACT enabled.
================================================================================

Tokenizer vocab: 268 | Dataset samples: 7842

Adapter configuration:
  dataset_path: datasets\styles.jsonl
  formatter_key: styles
  dataset_fraction: 0.6
  dataset_cap: None
  max_length: 896
  batch_size: 4
  num_workers: 27
  seed: 42
  epochs_per_domain: 1
  domain_cycles: 6
  steps_per_domain: 150
  learning_rate: 5e-05
  weight_decay: 0.01
  gradient_clip: 1.0
  device: cuda
  hpf_enable: True
  hpf_lowrank_ratio: 0.15
  pncec_enable: True
  pncec_cycle_length: 40
  pncec_compression_scale: 0.75
  pncec_expansion_scale: 1.25
  akap_enable: True
  domain_names: ('syntactic', 'semantic', 'reasoning')
  act_enable: True
  target_memory_utilization: 0.7
  checkpoint_dir: checkpoints\lonpt
  log_every: 25

Starting LONPT run...

=== LONPT Cycle 1/6 ===

[AKAP] Cycle 1: domain 'semantic' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'semantic' -> 8
[Domain:semantic] Epoch 1/1
  Step 0000 | phase=compression | lr_scale=0.75 | loss=5.5910
  Step 0025 | phase=  stabilize | lr_scale=1.00 | loss=4.3570
  Step 0050 | phase=  expansion | lr_scale=1.25 | loss=3.8387
  Step 0075 | phase=     settle | lr_scale=1.00 | loss=3.5856
  Step 0100 | phase=compression | lr_scale=0.75 | loss=3.4382
  Step 0125 | phase=  stabilize | lr_scale=1.00 | loss=3.4513
[Checkpoint] Saved lonpt_semantic_000150.pt (val_loss=3.5209)
[Domain:semantic] Validation loss: 3.5209 (✓ BEST)

[AKAP] Cycle 1: domain 'reasoning' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'reasoning' -> 8
[Domain:reasoning] Epoch 1/1
  Step 0000 | phase=  expansion | lr_scale=1.25 | loss=3.4117
  Step 0025 | phase=     settle | lr_scale=1.00 | loss=3.5672
  Step 0050 | phase=compression | lr_scale=0.75 | loss=3.6586
  Step 0075 | phase=  stabilize | lr_scale=1.00 | loss=3.6669
  Step 0100 | phase=  expansion | lr_scale=1.25 | loss=3.4241
  Step 0125 | phase=     settle | lr_scale=1.00 | loss=3.6168
[Checkpoint] Saved lonpt_reasoning_000300.pt (val_loss=3.5181)
[Domain:reasoning] Validation loss: 3.5181 (✓ BEST)

[AKAP] Cycle 1: domain 'syntactic' (~1.41M tokens)
[ACT] Adjusted batch size for domain 'syntactic' -> 8
[Domain:syntactic] Epoch 1/1
  Step 0000 | phase=compression | lr_scale=0.75 | loss=3.5500
  Step 0025 | phase=  stabilize | lr_scale=1.00 | loss=3.3775
  Step 0050 | phase=  expansion | lr_scale=1.25 | loss=3.6577
  Step 0075 | phase=     settle | lr_scale=1.00 | loss=3.3846
  Step 0100 | phase=compression | lr_scale=0.75 | loss=3.5499
  Step 0125 | phase=  stabilize | lr_scale=1.00 | loss=3.4133
[Domain:syntactic] Validation loss: 3.5689 (3.5689 (no improvement))

=== LONPT Cycle 2/6 ===

[AKAP] Cycle 2: domain 'semantic' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'semantic' -> 8
[Domain:semantic] Epoch 1/1
  Step 0000 | phase=  expansion | lr_scale=1.25 | loss=3.4569
  Step 0025 | phase=     settle | lr_scale=1.00 | loss=3.7738
  Step 0050 | phase=compression | lr_scale=0.75 | loss=3.4309
  Step 0075 | phase=  stabilize | lr_scale=1.00 | loss=3.4247
  Step 0100 | phase=  expansion | lr_scale=1.25 | loss=3.5264
  Step 0125 | phase=     settle | lr_scale=1.00 | loss=3.6373
[Checkpoint] Saved lonpt_semantic_000600.pt (val_loss=3.5069)
[Domain:semantic] Validation loss: 3.5069 (✓ BEST)

[AKAP] Cycle 2: domain 'reasoning' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'reasoning' -> 8
[Domain:reasoning] Epoch 1/1
  Step 0000 | phase=compression | lr_scale=0.75 | loss=3.5468
  Step 0025 | phase=  stabilize | lr_scale=1.00 | loss=3.8986
  Step 0050 | phase=  expansion | lr_scale=1.25 | loss=3.5429
  Step 0075 | phase=     settle | lr_scale=1.00 | loss=3.7271
  Step 0100 | phase=compression | lr_scale=0.75 | loss=3.3646
  Step 0125 | phase=  stabilize | lr_scale=1.00 | loss=3.4992
[Domain:reasoning] Validation loss: 3.5105 (3.5105 (no improvement))

[AKAP] Cycle 2: domain 'syntactic' (~1.41M tokens)
[ACT] Adjusted batch size for domain 'syntactic' -> 8
[Domain:syntactic] Epoch 1/1
  Step 0000 | phase=  expansion | lr_scale=1.25 | loss=3.9695
  Step 0025 | phase=     settle | lr_scale=1.00 | loss=5.0170
  Step 0050 | phase=compression | lr_scale=0.75 | loss=3.6925
  Step 0075 | phase=  stabilize | lr_scale=1.00 | loss=3.7334
  Step 0100 | phase=  expansion | lr_scale=1.25 | loss=3.4432
  Step 0125 | phase=     settle | lr_scale=1.00 | loss=3.4614
[Domain:syntactic] Validation loss: 3.5680 (3.5680 (no improvement))

=== LONPT Cycle 3/6 ===

[AKAP] Cycle 3: domain 'semantic' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'semantic' -> 8
[Domain:semantic] Epoch 1/1
  Step 0000 | phase=compression | lr_scale=0.75 | loss=3.4501
  Step 0025 | phase=  stabilize | lr_scale=1.00 | loss=3.4718
  Step 0050 | phase=  expansion | lr_scale=1.25 | loss=3.5080
  Step 0075 | phase=     settle | lr_scale=1.00 | loss=3.6635
  Step 0100 | phase=compression | lr_scale=0.75 | loss=3.6236
  Step 0125 | phase=  stabilize | lr_scale=1.00 | loss=3.4046
[Checkpoint] Saved lonpt_semantic_001050.pt (val_loss=3.2737)
[Domain:semantic] Validation loss: 3.2737 (✓ BEST)

[AKAP] Cycle 3: domain 'reasoning' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'reasoning' -> 8
[Domain:reasoning] Epoch 1/1
  Step 0000 | phase=  expansion | lr_scale=1.25 | loss=3.1963
  Step 0025 | phase=     settle | lr_scale=1.00 | loss=3.1768
  Step 0050 | phase=compression | lr_scale=0.75 | loss=3.2738
  Step 0075 | phase=  stabilize | lr_scale=1.00 | loss=3.0943
  Step 0100 | phase=  expansion | lr_scale=1.25 | loss=3.2399
  Step 0125 | phase=     settle | lr_scale=1.00 | loss=3.1351
[Checkpoint] Saved lonpt_reasoning_001200.pt (val_loss=3.1635)
[Domain:reasoning] Validation loss: 3.1635 (✓ BEST)

[AKAP] Cycle 3: domain 'syntactic' (~1.41M tokens)
[ACT] Adjusted batch size for domain 'syntactic' -> 8
[Domain:syntactic] Epoch 1/1
  Step 0000 | phase=compression | lr_scale=0.75 | loss=3.1679
  Step 0025 | phase=  stabilize | lr_scale=1.00 | loss=3.0744
  Step 0050 | phase=  expansion | lr_scale=1.25 | loss=3.1300
  Step 0075 | phase=     settle | lr_scale=1.00 | loss=3.0461
  Step 0100 | phase=compression | lr_scale=0.75 | loss=3.0159
  Step 0125 | phase=  stabilize | lr_scale=1.00 | loss=3.0247
[Checkpoint] Saved lonpt_syntactic_001350.pt (val_loss=2.9779)
[Domain:syntactic] Validation loss: 2.9779 (✓ BEST)

=== LONPT Cycle 4/6 ===

[AKAP] Cycle 4: domain 'semantic' (~1.40M tokens)
[Domain:semantic] Epoch 1/1
  Step 0000 | phase=  expansion | lr_scale=1.25 | loss=3.1443
  Step 0025 | phase=     settle | lr_scale=1.00 | loss=2.8883
  Step 0050 | phase=compression | lr_scale=0.75 | loss=2.9439
  Step 0075 | phase=  stabilize | lr_scale=1.00 | loss=3.0539
  Step 0100 | phase=  expansion | lr_scale=1.25 | loss=2.7836
  Step 0125 | phase=     settle | lr_scale=1.00 | loss=2.7967
[Checkpoint] Saved lonpt_semantic_001500.pt (val_loss=2.8482)
[Domain:semantic] Validation loss: 2.8482 (✓ BEST)

[AKAP] Cycle 4: domain 'reasoning' (~1.40M tokens)
[Domain:reasoning] Epoch 1/1
  Step 0000 | phase=compression | lr_scale=0.75 | loss=3.0469
  Step 0025 | phase=  stabilize | lr_scale=1.00 | loss=3.1271
  Step 0050 | phase=  expansion | lr_scale=1.25 | loss=2.8513
  Step 0075 | phase=     settle | lr_scale=1.00 | loss=2.9035
  Step 0100 | phase=compression | lr_scale=0.75 | loss=2.7464
  Step 0125 | phase=  stabilize | lr_scale=1.00 | loss=2.9192
[Checkpoint] Saved lonpt_reasoning_001650.pt (val_loss=2.8189)
[Domain:reasoning] Validation loss: 2.8189 (✓ BEST)

[AKAP] Cycle 4: domain 'syntactic' (~1.41M tokens)
[Domain:syntactic] Epoch 1/1
  Step 0000 | phase=  expansion | lr_scale=1.25 | loss=2.7436
  Step 0025 | phase=     settle | lr_scale=1.00 | loss=2.9211
  Step 0050 | phase=compression | lr_scale=0.75 | loss=2.9791
  Step 0075 | phase=  stabilize | lr_scale=1.00 | loss=2.8679
  Step 0100 | phase=  expansion | lr_scale=1.25 | loss=2.6751
  Step 0125 | phase=     settle | lr_scale=1.00 | loss=2.6366
[Checkpoint] Saved lonpt_syntactic_001800.pt (val_loss=2.7918)
[Domain:syntactic] Validation loss: 2.7918 (✓ BEST)

=== LONPT Cycle 5/6 ===

[AKAP] Cycle 5: domain 'semantic' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'semantic' -> 8
[Domain:semantic] Epoch 1/1
  Step 0000 | phase=compression | lr_scale=0.75 | loss=2.6805
  Step 0025 | phase=  stabilize | lr_scale=1.00 | loss=2.8304
  Step 0050 | phase=  expansion | lr_scale=1.25 | loss=2.8221
  Step 0075 | phase=     settle | lr_scale=1.00 | loss=2.6696
  Step 0100 | phase=compression | lr_scale=0.75 | loss=2.9712
  Step 0125 | phase=  stabilize | lr_scale=1.00 | loss=2.8166
[Checkpoint] Saved lonpt_semantic_001950.pt (val_loss=2.7058)
[Domain:semantic] Validation loss: 2.7058 (✓ BEST)

[AKAP] Cycle 5: domain 'reasoning' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'reasoning' -> 8
[Domain:reasoning] Epoch 1/1
  Step 0000 | phase=  expansion | lr_scale=1.25 | loss=2.7594
  Step 0025 | phase=     settle | lr_scale=1.00 | loss=2.6883
  Step 0050 | phase=compression | lr_scale=0.75 | loss=2.6890
  Step 0075 | phase=  stabilize | lr_scale=1.00 | loss=2.6787
  Step 0100 | phase=  expansion | lr_scale=1.25 | loss=2.7635
  Step 0125 | phase=     settle | lr_scale=1.00 | loss=2.7635
[Domain:reasoning] Validation loss: 2.7110 (2.7110 (no improvement))

[AKAP] Cycle 5: domain 'syntactic' (~1.41M tokens)
[ACT] Adjusted batch size for domain 'syntactic' -> 8
[Domain:syntactic] Epoch 1/1
  Step 0000 | phase=compression | lr_scale=0.75 | loss=2.7673
  Step 0025 | phase=  stabilize | lr_scale=1.00 | loss=2.7731
  Step 0050 | phase=  expansion | lr_scale=1.25 | loss=2.7372
  Step 0075 | phase=     settle | lr_scale=1.00 | loss=2.7088
  Step 0100 | phase=compression | lr_scale=0.75 | loss=2.7486
  Step 0125 | phase=  stabilize | lr_scale=1.00 | loss=2.6203
[Checkpoint] Saved lonpt_syntactic_002250.pt (val_loss=2.6863)
[Domain:syntactic] Validation loss: 2.6863 (✓ BEST)

=== LONPT Cycle 6/6 ===

[AKAP] Cycle 6: domain 'semantic' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'semantic' -> 8
[Domain:semantic] Epoch 1/1
  Step 0000 | phase=  expansion | lr_scale=1.25 | loss=2.6186
  Step 0025 | phase=     settle | lr_scale=1.00 | loss=2.6252
  Step 0050 | phase=compression | lr_scale=0.75 | loss=2.7047
  Step 0075 | phase=  stabilize | lr_scale=1.00 | loss=2.6791
  Step 0100 | phase=  expansion | lr_scale=1.25 | loss=2.7343
  Step 0125 | phase=     settle | lr_scale=1.00 | loss=2.5699
[Checkpoint] Saved lonpt_semantic_002400.pt (val_loss=2.6489)
[Domain:semantic] Validation loss: 2.6489 (✓ BEST)

[AKAP] Cycle 6: domain 'reasoning' (~1.40M tokens)
[ACT] Adjusted batch size for domain 'reasoning' -> 8
[Domain:reasoning] Epoch 1/1
  Step 0000 | phase=compression | lr_scale=0.75 | loss=2.5723
  Step 0025 | phase=  stabilize | lr_scale=1.00 | loss=2.7324
  Step 0050 | phase=  expansion | lr_scale=1.25 | loss=2.5500
  Step 0075 | phase=     settle | lr_scale=1.00 | loss=2.7229
  Step 0100 | phase=compression | lr_scale=0.75 | loss=2.5974
  Step 0125 | phase=  stabilize | lr_scale=1.00 | loss=2.6402
[Domain:reasoning] Validation loss: 2.6790 (2.6790 (no improvement))

[AKAP] Cycle 6: domain 'syntactic' (~1.41M tokens)
[ACT] Adjusted batch size for domain 'syntactic' -> 8
[Domain:syntactic] Epoch 1/1
  Step 0000 | phase=  expansion | lr_scale=1.25 | loss=2.7159
  Step 0025 | phase=     settle | lr_scale=1.00 | loss=2.6905
  Step 0050 | phase=compression | lr_scale=0.75 | loss=2.6184
  Step 0075 | phase=  stabilize | lr_scale=1.00 | loss=2.6067
  Step 0100 | phase=  expansion | lr_scale=1.25 | loss=2.7251
  Step 0125 | phase=     settle | lr_scale=1.00 | loss=2.6497
[Checkpoint] Saved lonpt_syntactic_002700.pt (val_loss=2.6459)
[Domain:syntactic] Validation loss: 2.6459 (✓ BEST)

Done! Checkpoints saved to: checkpoints\lonpt
(.venv) PS C:\Users\trent\aerin>