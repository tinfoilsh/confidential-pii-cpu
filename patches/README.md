# Patches applied to installed packages at image build

`-p1` unified diffs rooted at `/`, targeting the installed site-packages
paths (same convention as confidential-glm5-3-flash). Applied by the
Dockerfile after `pip install`.

- `0001-cpu-moe-grouped-matmul.patch` — opf's CPU MoE fallback gathered
  (copied) the chosen experts' full weight matrices per 32-token chunk per
  layer; this ports the repo's own Triton algorithm (group tokens by
  expert, weights read in place, one index_add_) to plain torch.
  Measured: 99x on a 600-token forward pass, max logit diff 1e-5.
  Proposed upstream to openai/privacy-filter; drop this patch and bump the
  requirements.txt pin once merged.
- `0002-runtime-expose-forward-and-decode-stages.patch` — splits
  `opf/_core/runtime.py::predict_text` into reusable stages (`encode_text`,
  `forward_window`, `decode_token_scores`) with no behaviour change, so
  `server.py` can run the window forwards elsewhere (the gateway's fair
  queue across workers) and feed the stitched scores to the unchanged global
  decode. Regenerate with `git diff` against the pristine tarball with
  0001 applied.
