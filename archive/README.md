# Archive — superseded scripts

The four original analysis scripts, kept for provenance. **Do not use these to
regenerate figures.** They are superseded by `src/hipo/`.

Known issues, each documented in `docs/ADDENDUM_CONSISTENCY.md` and the code review:

| Script | Issue |
|---|---|
| `polymer_conformation_V1_flip.py`, `V2_flip.py` | `unnorm_pairwise_joint` uses `np.outer(...)`, forcing I(φᵢ;φⱼ) = 0 in every unnormalised joint panel. Orientation average reads both directions at the same index instead of the mirrored one. float32 at the JAX boundary. V1 and V2 differ only in 2-D display orientation. |
| `nmer_anchor_phirest_joint_flip.py` | `anchor="first"` applies one transition too many before the accumulator, so the first angle folded into Φ_rest belongs to dimer 3 rather than 2. `anchor="last"` is correct. |
| `cis_trans_profile_flip5_updated.py` | Correct as far as tested; its mirrored orientation-average convention is the one carried into `src/hipo/core.py`. |
