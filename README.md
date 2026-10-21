# NCM Forge

[![CI](https://github.com/kavai-no/ncm-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/kavai-no/ncm-forge/actions/workflows/ci.yml)

Next-concept modeling: https://github.com/HazyResearch/ncm

Train a next-concept model on text from a regular Hugging Face model, train one from scratch, or keep the token model and use it to repair the concept draft.

A next-concept model writes one level of discrete codes at a time, and every code in that level in one step. The token model does not become an NCM. It supplies the training text and the input embedding. At inference it rewrites only the draft tokens it does not trust. That repair is one teacher forward, not one forward per token.

The method is [Next-Concept Modeling](https://arxiv.org/abs/2610.14162) ([HazyResearch/ncm](https://github.com/HazyResearch/ncm)). This repository is an independent implementation. It does not include that code, and it does not copy a Qwen, Llama, or Gemma checkpoint into an NCM. Those parameterizations do not match.

## Modes

| Mode | What is trained | What runs at inference |
| --- | --- | --- |
| Distill | A multiscale residual VQ and a next-concept generator, on tokens sampled from the teacher | The concept model |
| Scratch | The same two stages, on the text you provide, with no teacher | The concept model |
| Hybrid | The distilled concept model, with the teacher kept | A concept draft, then one teacher forward that replaces low-confidence tokens |

## What you can run

| Piece | Where | What it does |
| --- | --- | --- |
| Browser lab | this app | Distill, train from scratch, or combine both on a tiny corpus, with a live ladder and samples |
| Train notebook | [`src/lib/ncm/ncm_hybrid_colab.ipynb`](src/lib/ncm/ncm_hybrid_colab.ipynb) | The same recipe on a free Colab T4 (16 GB), trained on CodeSearchNet Python functions. Default teacher is `Qwen/Qwen2.5-0.5B-Instruct`. `Qwen2.5-1.5B-Instruct` also fits. 3B and larger are refused |
| Test notebook | [`src/lib/ncm/ncm_hybrid_test.ipynb`](src/lib/ncm/ncm_hybrid_test.ipynb) | Loads `hybrid.pt` and scores reconstruction, next-level codes, and the hybrid repair on text the training run did not use |

## What a short run will not do

The lab is a miniature. The Colab notebook freezes a Qwen-vocabulary tokenizer and trains a new generator for one pass over 131,072 CodeSearchNet windows. It stops when the held-out coarse continuation stops rising, and it does not keep training a generator that has memorized a small buffer. It is still not the paper's training job (billions of tokens). It will not reproduce the LM1B, OpenWebText, or XSum numbers. Judge a run by the held-out coarse continuation and by whether the continuation contains code, not by the training coarse accuracy or by teacher NLL.

Teacher weights stay under their own licenses. The MIT license in this repository covers this code only.

## Requirements

- Lab: Node.js 22.
- Notebooks: a Colab runtime with a T4 GPU (16 GB). The notebook installs PyTorch and Transformers. No local GPU is required.

## Develop

```bash
npm install
npm run dev
npm test
npm run typecheck
npm run build
```

`npm run dev` serves the lab. In the export card, **Train notebook** and **Test notebook** download the Colab files. The selected teacher is written into the train notebook when it fits a 16 GB T4. Larger picks stay on Qwen2.5-0.5B there.

## Colab

1. Runtime → Change runtime type → T4 GPU.
2. Run the train notebook from the top. Training windows are real Python functions from CodeSearchNet, not a prompt list. It writes `/content/ncm-hybrid/hybrid.pt`.
3. Run the test notebook. It looks for that file, then Drive, then asks for an upload.

The teacher is loaded in fp16 and unloaded before the concept model trains. A 0.5B run stays under about 8 GB. A 1.5B run stays under about 12 GB.

Notebooks are generated. Change [`scripts/build_colab_notebook.py`](scripts/build_colab_notebook.py) and [`scripts/build_test_notebook.py`](scripts/build_test_notebook.py), then regenerate. Do not edit the notebooks and the builders in opposite directions.

## Layout

```text
src/lib/ncm/engine.ts                 browser training loop
src/lib/ncm/python.ts                 exported Python trainer
src/lib/ncm/ncm_hybrid_colab.ipynb    train and benchmark
src/lib/ncm/ncm_hybrid_test.ipynb     load hybrid.pt and score it
src/components/forge-app.tsx          lab UI
```

## Citation

The method, not this repository:

```bibtex
@article{kumbong2026ncm,
  title   = {Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation},
  author  = {Kumbong, Hermann and Khare, Ishan S. and Mlauzi, Iddah and Woen, Andrew and Ding, Charles and Buchanan, E. Kelly and Arora, Simran and Mirhoseini, Azalia and R{\'e}, Christopher},
  journal = {arXiv preprint arXiv:2610.14162},
  year    = {2026},
  url     = {https://github.com/HazyResearch/ncm}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go through [SECURITY.md](SECURITY.md), not a public issue.

## License

[MIT](LICENSE), the same terms as [HazyResearch/ncm](https://github.com/HazyResearch/ncm). That license has no noncommercial or patent addendum. Their copyright stays on their code. See [NOTICE](NOTICE).
