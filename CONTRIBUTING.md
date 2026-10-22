# Contributing

## Before you open a pull request

1. Branch from `main`.
2. Run `npm test` and `npm run typecheck`.
3. Describe what changed and how you checked it. For a drafting change, say whether the text still matched Qwen and what happened to tokens per Qwen forward.

## What belongs here

- Fixes to the accept/reject rule, the lab, or the Colab notebooks.
- Tests that fail when speculative text differs from greedy text.

Do not commit secrets, `.env` files, or checkpoints. `drafter.pt` is a training output, not source.

## Notebooks

`scripts/build_speculative_notebook.py` writes `src/lib/ncm/qwen_speculative_colab.ipynb` and `src/lib/ncm/qwen_speculative_test.ipynb`.

Change the builder, then regenerate. Do not hand-edit the notebooks and the builder in opposite directions.

## License

By contributing, you agree that your contributions are licensed under the MIT License, the same terms as the rest of this repository.
