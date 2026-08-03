# BrepARG Same-Data VQ-VAE: Short Run Targeting 160 Epochs

This local comparison trained the original BrepARG SE-VQ-VAE with a target of 160 epochs, batch size 128, and learning rate `1e-4`. It used 606,160 training patches and 59,922 validation patches. The run did not complete 160 epochs: the recorded training ends at epoch 73 after disk exhaustion interrupted checkpoint writing.

The validation reconstruction loss improved from `0.00249944` at epoch 1 to `0.00023836` at epoch 73. The epoch-73 checkpoint was not retained successfully; the surviving best checkpoint is epoch 70 with validation reconstruction loss about `0.00024037`. Its external SHA-256 is recorded in `summary.json`, but the 660 MiB checkpoint is not stored in Git.

Files in this package:

- `events.out.tfevents.1784363313.DESKTOP-VDBMPQG.14760.0`: original TensorBoard event, epochs 1-73.
- `scalars.csv`: all 12 recorded scalar tags exported for every epoch.
- `summary.json`: structured training status, selected metrics, and external checkpoint identity.

Do not describe this as a completed 160-epoch result. The accurate label is "target 160, interrupted after epoch 73, retained best epoch 70."
