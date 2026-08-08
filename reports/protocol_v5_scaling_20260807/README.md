# Protocol V5 scaling evidence

This directory contains the lightweight, version-controlled evidence for the Protocol V5 scaling ladder completed on 2026-08-07. It intentionally excludes raw datasets, materialized protocol members, TensorBoard event files, and model checkpoints.

The JSON sweep manifests preserve train and validation sampling, parent coverage, epoch gates, bucket metrics, and checkpoint paths. `checkpoint_sha256.tsv` records the SHA-256, byte count, and original local path for each excluded checkpoint. The `analysis/` files contain the scaling decision and CSV points. The `logs/` files are copied as text for the two continuous-bypass oracle runs.

Decision: `CONTINUE_CAPACITY_INVESTIGATION`; the continuous bypass oracle completed, but the protocol does not authorize AR. The next experiment is the 100-200 CAD assembly calibration oracle.
