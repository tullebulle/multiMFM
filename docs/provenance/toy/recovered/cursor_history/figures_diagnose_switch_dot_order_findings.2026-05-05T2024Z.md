ON inverse ordering: Keeps samples in distributions! Neat result. Even with strong guidance towards it, the base distribution will control.



Main finding: yes, your intuition is right. The position dynamics are being pulled back by the base model toward the original French-flag color-position coupling.

  For lambda=0.35:

  t      |base_y| |guid_y| base->target guid->target total->target |base_c| |guid_c|
  0.500  0.776   1.235    0.480        0.258        0.738         1.383   0.658
  0.699  0.704   1.881   -0.225        0.640        0.415         1.665   0.289
  0.849  0.544   1.350   -0.426        0.220       -0.206         1.830   0.025
  0.982  0.660   0.448   -0.646        0.104       -0.542         1.840   0.010

  Interpretation:

  - The color-gradient direction is not consistently aligned with the swapped target color. The total color direction matches the nearest-dot target only about 20% late in the trajectory.

  Switch behavior:
  mean nearest-dot-target switches = 0.730
  max nearest-dot-target switches = 6
  fraction with any head-color switch = 0.247
  fraction with any nearest-dot switch = 0.523

  So individual samples do sometimes jump or switch assignment, but not in a clean controlled way. The more common failure mode is: guidance nudges the point toward a swapped dot, then the learned
  French prior reasserts itself and pulls the position/color pair back into a compatible original stripe.

  I added the diagnostic script at experiment_dot_guidance_diagnostics.py.





EXPERIMENT:

- Show robustness of it.
- Adding noised samples to your training data to get a bigger support of OOD for GLASS posterior sampling.