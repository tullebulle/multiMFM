# results/dna/diversity/regen -- Tables 4, 5 and the App. B.1 clustering statistic

Recomputed by `dmfm.experiments.analyze_c0_diversity` (the ported rebuttal analyser,
`bash scripts/dna/tables4_5_c0_diversity.sh`) on the parent-disjoint n = 1,000 C0 sets in
`results/dna/c0_guidance/n1000/` -- the same sample sets that `results/dna/regen/table6`
scores with Evo 2, so Tables 4, 5 and 6 are internally consistent.

The published Tables 4-6 were computed on the purged `split65k` pipeline; the surviving
record of those numbers stays in `results/dna/june_split65k_recovered/`, and the round-1
collection of this same recompute stays in `results/dna/diversity/n1000/`. Nothing there is
overwritten.

File names follow the paper: the analyser's `table6_baselines` is the paper's **Table 4**
and its `table7_generated` is the paper's **Table 5** (it predates the renumbering).

`notebooks/09_dna_regenerated.ipynb` shows both versions side by side.
