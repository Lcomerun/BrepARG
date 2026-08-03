# BrepARG Same-Data VQ-VAE: 400-Epoch Long Run

This run resumed the original BrepARG same-data SE-VQ-VAE from the complete epoch-70 model, optimizer, scaler, iteration, and best-loss state. It trained epochs 71 through 400 with batch size 128 and learning rate `1e-4`, and completed the target epoch.

The best validation reconstruction loss was `0.00021055` at epoch 269. At epoch 400, train reconstruction loss was `0.00002662`, validation reconstruction loss was `0.00021944`, validation codebook activity was `0.46289`, and validation perplexity was `150.19`. The continued reduction in train loss after epoch 269 did not improve the best validation reconstruction value.

Files in this package:

- `events.out.tfevents.1784547631.DESKTOP-VDBMPQG.27412.0`: original TensorBoard event, epochs 71-400.
- `scalars.csv`: all 12 recorded scalar tags exported for every continuation epoch.
- `summary.json`: structured final and best metrics plus external checkpoint hashes.

The best and epoch-400 checkpoints are each approximately 660 MiB and remain outside Git. Their SHA-256 identities are recorded in `summary.json` so copies on other machines can be verified.
