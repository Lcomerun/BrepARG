# Protocol V6 continuation assessment

Assessment time: 2026-08-13 10:50 +08:00.

## Decision

Further training is required only if the five-seed comparison remains the
acceptance target. The interrupted launcher must not be restarted unchanged.

The host was restarted by a planned Windows update while seed 3 learned VQ was
between epochs 43 and 44. The training stderr is empty. The rolling checkpoint
contains model weights and validation metadata, but it does not contain the
optimizer, AMP scaler, scheduler, or RNG state. Continuing from that file would
therefore not be strictly equivalent to the other fixed 100-epoch runs.

## Evidence available now

- Seeds 0, 1, and 2 completed all four 100-epoch loops.
- Seed 3 completed both FSQ loops, but each became numerically unstable.
- Seed 3 learned VQ has 44 fully finite epochs (0 through 43); its best
  validation reconstruction is `6.6613e-4` at epoch 40.
- Seed 3 continuous bypass did not start.
- Seed 4 did not start.
- Surface reconstruction did not start.

The healthy learned-VQ seeds reached their best values around epochs 95 and 96,
so the seed 3 epoch-43 result cannot stand in for a complete run. Conversely,
four independent seeds now show that both FSQ configurations become non-finite
under this training configuration. More identical FSQ compute is unlikely to
change the representation decision unless a complete five-seed matrix is
required for reporting.

## Recommended recovery scope

1. Preserve the interrupted seed 3 directory as evidence; do not overwrite it.
2. Start a new recovery output directory and train seed 3 learned VQ from
   initialization for 100 epochs, followed by seed 3 continuous bypass for 100
   epochs.
3. Train seed 4 learned VQ and continuous bypass for 100 epochs.
4. Run seed 4 FSQ only if strict completion of the original four-arm matrix is
   still required. It should not be treated as a likely promotion candidate
   without first fixing the recurrent non-finite behavior.
5. Run the fixed 100-CAD surface reconstruction only after the selected recovery
   matrix is complete, and keep sequence regeneration and AR blocked.

Restarting the existing launcher would skip seeds 0-2 but rerun all four seed 3
arms because `seed3/vqvae_hp_sweep.json` was never produced. It would also
overwrite seed 3 logs. A recovery launcher must select arms explicitly and use
a new output directory.
