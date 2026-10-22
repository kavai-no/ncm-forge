# NCM Forge

[![CI](https://github.com/kavai-no/ncm-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/kavai-no/ncm-forge/actions/workflows/ci.yml)

Next-concept modeling: https://github.com/HazyResearch/ncm

A small drafter proposes the next few tokens. Qwen checks that span in one forward and keeps only the prefix it would have written itself. The output is greedy Qwen. It is faster only when several tokens survive each check.

The cheap proposal is the idea in [Next-Concept Modeling](https://arxiv.org/abs/2610.14162) ([HazyResearch/ncm](https://github.com/HazyResearch/ncm)): do less work per token, then spend a check on the candidates. This repository does not copy that code. A codebook is not in the draft. A code that cannot say Qwen's token cannot be a proposal.

## What you can run

| Piece | Where | What it does |
| --- | --- | --- |
| Rule | this app | A fixed function, a drafter that agrees with Qwen some of the time, and the accept/reject trace. The text matches greedy Qwen, or the page says it does not |
| Train notebook | [`src/lib/ncm/qwen_speculative_colab.ipynb`](src/lib/ncm/qwen_speculative_colab.ipynb) | Trains a 4-layer drafter to copy `Qwen/Qwen2.5-0.5B-Instruct` on CodeSearchNet Python, on a free Colab T4 |
| Test notebook | [`src/lib/ncm/qwen_speculative_test.ipynb`](src/lib/ncm/qwen_speculative_test.ipynb) | Loads `drafter.pt` and checks that the continuation is greedy Qwen |

## What a short run will not do

The drafter sees a few thousand functions, not the data Qwen was trained on. If it agrees with Qwen on about one token per forward, there is no speedup and the notebook says so. The text is still Qwen's. A run that changes a token is a failed check, not a faster model.

Teacher weights stay under their own licenses. The MIT license in this repository covers this code only.

## Requirements

- Lab: Node.js 22.
- Notebooks: a Colab runtime with a T4 GPU (16 GB). The notebook installs PyTorch and Transformers.

## Develop

```bash
npm install
npm run dev
npm test
npm run typecheck
npm run build
```

`npm run dev` serves the lab. **Train notebook** and **Test notebook** download the Colab files.

## Colab

1. Runtime → Change runtime type → T4 GPU.
2. Run the train notebook from the top. It writes `/content/qwen-draft/drafter.pt`.
3. Run the test notebook. It looks for that file, then Drive, then asks for an upload.

The teacher is loaded in fp16 to label the training text, then unloaded. It is loaded again for the check. The drafter trains on Qwen's next-token labels, not on a codebook.

Notebooks are generated. Change [`scripts/build_speculative_notebook.py`](scripts/build_speculative_notebook.py), then regenerate. Do not edit the notebooks and the builder in opposite directions.

## Layout

```text
src/lib/ncm/speculate.ts                         the accept/reject rule
src/lib/ncm/qwen_speculative_colab.ipynb         train the drafter and check it
src/lib/ncm/qwen_speculative_test.ipynb          load drafter.pt and check it
src/components/forge-app.tsx                     lab UI
```

## Citation

The method the proposal is taken from, not this repository:

```bibtex
@article{kumbong2026ncm,
  title   = {Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation},
  author  = {Kumbong, Hermann and Khare, Ishan S. and Mlauzi, Iddah and Woen, Andrew and Ding, Charles and Buchanan, E. Kelly and Arora, Simran and Mirhoseini, Azalia and R{\'e}, Christopher},
  journal = {arXiv preprint arXiv:2610.14162},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.14162}
}
```
