# ADR-0001: Preserve CAD topology for boundary-consistency training

## Status

Accepted for implementation only after the P0-A and P0-B gates pass.

## Date

2026-08-13

## Context

The proposed boundary-consistency experiment must compare a decoded shared edge with the two decoded faces incident to that edge after all three patches are mapped from normalized coordinate space (NCS) to world coordinate space (WCS). The current representation-training pipeline cannot identify that relation. `breparg_improvements/vqvae_sampling.py` splits each parsed CAD into independent patch records, drops bounding boxes and adjacency, and then content-deduplicates patches. Geometrically identical patches from different CADs can therefore share one training record even though their WCS placement and neighbors differ.

The stored surface grid is also a coarse 32-by-32 UV sample without an explicit trim-boundary mask. On the frozen 100-CAD cohort, ground-truth shared edges have a non-zero nearest-surface squared-distance floor. A raw Chamfer penalty would encourage otherwise correct geometry to move toward sampling-grid points.

Learned VQ adds another constraint: a quantizer forward in training mode updates codebook usage, embeddings, and its feature pool. Executing an auxiliary relation forward and multiplying its loss by zero would still change training state, so it is not a valid weight-zero control.

## Decision

If the P0 gates open, create a separate topology-preserving relation dataset from the original parsed per-CAD records. Each item binds one manifold edge to exactly two distinct incident faces and includes entity indices, NCS surface and edge patches, and their WCS bounding boxes. The builder validates `edgeFace_adj` and `faceEdge_adj` bidirectionally, rejects invalid or non-finite bounding boxes, keeps parent-isolated split membership, and does not apply cross-CAD content deduplication.

Use the inverse parser transform for every decoded entity:

    center = (bbox_min + bbox_max) / 2
    scale = max(bbox_max - bbox_min)
    points_wcs = points_ncs * scale / 2 + center

The first loss version uses one-way edge-to-surface nearest-neighbor squared distance computed with `torch.cdist`. For each incident face it penalizes only predicted distance above a stopped-gradient ground-truth distance baseline. It does not use the reverse surface-to-edge term, because that term would pull the entire surface toward the boundary.

Weight zero exits before relation-loader iteration, relation RNG consumption, or auxiliary quantizer execution. For positive weights, the auxiliary quantizer runs without training-time codebook side effects while gradients remain enabled through the encoder and decoder. The three weights are `0`, `0.1`, and `1.0`.

## Alternatives Considered

### Extend the deduplicated patch cache

Rejected because the cache no longer has a unique CAD placement, entity identity, or adjacency relation. Adding an arbitrary provenance after deduplication would silently train on the wrong face-edge pairing.

### Raw bidirectional Chamfer distance

Rejected because the ground-truth UV grid already has a non-zero edge-to-surface distance floor, and the surface-to-edge direction would attract interior surface samples toward the trim boundary.

### Multiply the auxiliary loss by zero for the control arm

Rejected because learned-VQ forward calls mutate codebook and feature-pool state. The control would differ even with a numerically zero weighted loss.

## Consequences

The boundary experiment requires a new relation data path and cannot be implemented as a one-line loss change. It adds topology validation and I/O cost, but preserves causal interpretation of the weight comparison. Weight zero can be tested for state-level equivalence with the existing training path. The loss is evaluated primarily by strict BRep validity on the frozen 100 CADs; reconstruction MSE and boundary distance remain explanatory metrics. If P0-B does not achieve zero non-finite events or the healthy VQ 100-CAD baseline is incomplete, this decision remains documented but unimplemented.
