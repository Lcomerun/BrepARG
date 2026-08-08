# Assembly calibration, 100 validation CAD

This report closes the first assembly-calibration decision experiment on 100 deterministic, parent-isolated Protocol V5 validation CAD records. Every CAD was evaluated with the same parsed topology under three arms: unmodified original patches, the continuous-bypass 64D checkpoint, and the FSQ-8192/4D checkpoint. All failures remain in the attempts denominator.

Strict BRep validity was 84/100 for original patches, 70/100 for continuous bypass, and 49/100 for FSQ. Of the 84 original-valid CADs, bypass preserved validity for 69 and lost it for 15. The curved-MSE separation between bypass valid and invalid CADs was modest and the binned validity curve was not monotonic, while CAD face and edge counts showed a stronger descriptive shift. The formal decision is `ASSEMBLY_DOMINATED`.

The next authorized work is assembly-chain repair and matched re-evaluation. Learned VQ-4096/64D at 300k and decoder surgery remain deferred until assembly failures are separated from representation failures. AR remains blocked.

`calibration_manifest.jsonl` contains 300 attempt rows. `calibration_state.json` binds the protocol and checkpoints. `analysis/` contains the formal summary and Pillow-rendered calibration plot. `logs/` contains text stdout/stderr. STEP files remain outside Git; `step_sha256.csv` binds the 284 retained local STEP files by relative path, byte count, and SHA-256. The remaining 16 attempts are assembly failures with no retained STEP.
