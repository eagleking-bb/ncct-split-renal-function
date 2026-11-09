"""Shared components for the GFR prediction training entrypoints.

The modules in this package exist so that the five deep-learning entrypoints
(``01``-``05``) and the external-inference entrypoint use one implementation of

* tabular imputation / standardisation (``preprocessing``), and
* the deterministic-cache plus per-epoch random-augmentation image pipeline
  (``monai_pipeline``).

Both are required for a consistent comparison between architectures and for a
checkpoint that can be re-used for inference without re-fitting anything.
"""
