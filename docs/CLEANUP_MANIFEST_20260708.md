# V13 Cleanup Manifest 2026-07-08

This manifest records how the workspace was simplified for a server restart. It is intentionally concise; detailed historical evidence remains in git history, paper materials, or regenerated local reports.

## Kept As Canonical

Project entry:

    README.md
    PROJECT_INDEX.md
    AGENTS.md
    PLANS.md

Server restart:

    docs/SERVER_START_HERE.md
    environment.server.yml
    tools/server_bootstrap.sh
    tools/run_vqvae_from_patch_shards.sh
    tools/build_server_package.ps1

Sharded data:

    docs/v13_sharded_dataset_operator_guide.md
    plans/v13_sharded_dataset_execplan.md
    tools/build_parsed_shards.py
    tools/verify_parsed_shards.py
    tools/build_vqvae_patch_shards.py
    tools/run_parsed_shard_cycle.py
    breparg_improvements/sharded_data.py
    breparg_improvements/vqvae_sampling.py

Current cleanup plan:

    plans/v13_workspace_cleanup_and_server_packaging_execplan.md

Paper source:

    papers/aaai_v13/README.md
    papers/aaai_v13/evidence_map.md
    papers/aaai_v13/latex/main.tex
    papers/aaai_v13/latex/supplement.tex
    papers/aaai_v13/latex/main.pdf
    papers/aaai_v13/latex/supplement.pdf
    papers/aaai_v13/supplement_staging/

## Retained Heavy Artifacts Outside The Source Package

Do not delete these until a verified server copy exists:

    C:\V13_abc_parsed_shards
    ABC/processed/abc_parsed_full_archives
    ABC/processed/train_outputs/newscheme_full_vqvae_epoch100
    local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6

The server package excludes these paths by design.

## Merged Into Canonical Docs

The useful facts from older local reports were merged into:

    README.md
    PROJECT_INDEX.md
    docs/SERVER_START_HERE.md
    docs/v13_sharded_dataset_operator_guide.md

Important merged facts:

    parsed shards are complete at C:\V13_abc_parsed_shards
    100 parsed shards are verified
    server should build VQ patch shards before VQ-VAE training
    VQ-VAE must run before source-path sequence rebuild and AR long-context training
    generated G20/G100 evidence remains diagnostic, not positive paper evidence

## Safe Deletion Class

These files are generated and safe to remove:

    __pycache__/
    .pytest_cache/
    tmp/
    zero-byte local report logs
    LaTeX .aux/.bbl/.blg/.log files
    historical PDF render checks under papers/aaai_v13/latex/rendered/

Canonical paper PDFs and source `.tex` files are not part of this deletion class.

## Ignore Policy

The `.gitignore` now hides generated local state:

    local_reports/
    tmp/
    dist/
    breparg_improvements/repro_outputs/
    papers/aaai_v13/latex/rendered/

This keeps future status output focused on source, docs, plans, and server scripts. Generated reports can still be written and uploaded when a tool explicitly needs them.

## Completion Notes

Actions completed in this pass:

    added root README.md and compact PROJECT_INDEX.md
    added docs/SERVER_START_HERE.md
    added environment.server.yml
    added tools/server_bootstrap.sh
    added tools/run_vqvae_from_patch_shards.sh
    added tools/build_server_package.ps1
    built dist/v13_server_ready_20260708.zip
    deleted tmp/
    deleted Python __pycache__/ and .pytest_cache/ after validation
    deleted selected sharding-cycle logs already summarized by the final shard manifest
    deleted historical paper render checks under papers/aaai_v13/latex/rendered/
    kept canonical main_page-* and supplement_page-* paper renders

Validation completed in this pass:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_local_pipeline_helpers
    result: 118 tests OK

    PYTHONUTF8=1 C:\Users\YU\.conda\envs\brepgen_env\python.exe breparg_improvements/test_all.py
    result: 58 passed, 0 failed

    D:\Program Files\Git\bin\bash.exe -n tools/server_bootstrap.sh tools/run_vqvae_from_patch_shards.sh tools/run_vqvae_complex_recovery.sh tools/run_source_path_sequence_rebuild.sh tools/run_ar_v13_long_context.sh
    result: exit code 0

Residual server-side checks:

    verify CUDA on the rented server
    verify OCC.Core.TopoDS in the rented-server environment
    verify uploaded parsed shards at /workspace/ABC/processed/abc_parsed_shards
    verify baseline checkpoint exists at /workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt
    run tools/run_vqvae_from_patch_shards.sh on the server and confirm the log says VQ patch-shard sampling selected=...

The final cleanup leaves the root readable and the server path executable from `docs/SERVER_START_HERE.md`. Any remaining untracked files after cleanup should be either newly created source/docs/scripts or deliberately retained heavy artifacts ignored by `.gitignore`.
