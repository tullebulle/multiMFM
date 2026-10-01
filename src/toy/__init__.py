"""multiMFM toy example (paper Fig. 1, Table 12, App. E.2 / F.7).

Coloured particles x = (c, y): a colour c in {Blue, White, Red} (one-hot in R^3,
model order B, W, R) and a position y in R^2.  The base distribution is
p1 = U(colours) x U([-1, 1]^2).  A small DiT flow model learns p1, a multimodal
Meta Flow Map (MFM) is distilled from its GLASS posterior velocity (diagonal
stage, then ESD consistency stage), and reward guidance steers samples into a
Norway flag (colour only) or a smiley (position + colour).

The original code (repo ``norway``) was purged.  Some modules here are
recovered verbatim from transcripts / editor history, the rest are
reconstructions; see ``docs/provenance/toy/README.md`` for which is which.

Modules
-------
data                   toy distributions (reconstructed)
model                  base multimodal DiT (recovered verbatim)
mfm_model              multimodal MFM (recovered verbatim, tail reconstructed)
glass                  GLASS posterior velocity (recovered verbatim)
value_fn               teacher velocity + GLASS posterior / value estimator (reconstructed)
mfm_value_fn           MFM posterior sampler / value estimator (reconstructed)
reward, reward_smiley  Norway / smiley rewards, App. F.7.2-3 (reconstructed)
train                  base flow-matching trainer (reconstructed)
train_mfm_diagonal     diagonal GLASS distillation (recovered, CLI reconstructed)
train_mfm_consistency  ESD consistency stage (partly recovered)
probe_mfm_diagonal     diagonal probe vs GLASS (recovered verbatim)
sample                 base sampler + checkpoint loading (reconstructed)
guidance               delayed-colour / hybrid guided samplers (reconstructed)
figure                 Fig. 1 four-panel plot (plotting code recovered from the
                       figure notebook's editor history)
fig1                   driver: runs every panel with both guidance backends and
                       writes results/toy/ (new)
table12                Table 12 paper values vs what the checkpoints record (new)
paths                  repo-relative default locations (new)
"""
