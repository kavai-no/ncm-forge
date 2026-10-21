# Security

## Reporting

Please do not open a public issue for a security problem.

Open a private advisory from the Security tab of this repository, with the version or commit, the impact, and a way to reproduce it if you have one.

We will acknowledge the report and tell you when a fix is on `main`.

## Scope

This repository trains and scores models. A checkpoint and the teacher weights are not secrets by themselves, but do not send private fine-tunes or access tokens with a report. Hugging Face tokens belong in Colab secrets, not in the notebook source.
