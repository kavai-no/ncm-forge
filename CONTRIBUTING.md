# Contributing

## Before you open a pull request

1. Branch from `main`.
2. Run `npm test` and `npm run typecheck`.
3. Describe what changed and how you checked it. For a training or scoring change, say what the numbers did, including when they got worse.

## What belongs here

- Fixes to the lab, the exported trainer, or the Colab notebooks.
- Tests that fail when the hybrid repair rule is wrong.

Do not commit secrets, `.env` files, or checkpoints. `hybrid.pt` is a training output, not source.

## Notebooks

`scripts/build_colab_notebook.py` writes `src/lib/ncm/ncm_hybrid_colab.ipynb`.
`scripts/build_test_notebook.py` copies the model cell from that notebook into `src/lib/ncm/ncm_hybrid_test.ipynb`.

Change the builders, then regenerate. Do not hand-edit the notebooks and the builders in opposite directions.

## License

By contributing, you agree that your contributions are licensed under the MIT License, the same terms as the rest of this repository.
