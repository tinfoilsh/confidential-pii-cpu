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
