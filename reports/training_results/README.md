# Curated Training Results

This directory stores small, reviewable training evidence copied or derived from external run directories. Each run package may include the original TensorBoard event file, a complete scalar CSV export, a machine-readable summary, and a human-readable interpretation. Checkpoints, datasets, token sequences, generated CAD, tqdm stderr streams, and other large run artifacts remain outside Git.

The current packages document the original BrepARG same-data VQ-VAE comparison training. The run called "short target160" was configured for 160 epochs but stopped during checkpoint writing after epoch 73; it must not be reported as a completed 160-epoch run. The long run restored the epoch-70 checkpoint and completed epochs 71 through 400.
