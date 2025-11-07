# The Annotated Discrete Diffusion Models

*An annotated implementation of a character-level disrete diffusion model for text generation.*

---

![Denoising Demo](./assets/text_diffusion.gif)  
*a character-level discrete diffusion model in action.*

---

## Overview

This repository contains a single, self-contained Jupyter Notebook that walks through the theory and implementation of **discrete diffusion models for text generation** inspired by the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834).

The notebook adapts **Andrej Karpathy’s character-level baby GPT**, a 7.23M parameter model, (from his [nanoGPT](https://github.com/karpathy/nanoGPT) repository) into a **discrete diffusion model** capable of learning to denoise corrupted text back into coherent sequences.

Unlike autoregressive models that generate text one token at a time, diffusion models generate by **denoising all tokens in parallel**, offering a powerful alternative paradigm for language modelling.

## Usage

Run the notebook in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ash80/diffusion-gpt/blob/master/The_Annotated_Discrete_Diffusion_Models.ipynb)

Or in your local Jupyter instance load the notebook and run the cells sequentially. Optionally, adjust dataset, noise schedule, or model size to experiment with your own text corpus.

## What is covered

* **Mathematical framework** of discrete diffusion models
* **Continuous-time Markov chain** formulation for token corruption
* Adaptation of **Karpathy’s baby GPT architecture** for character-level text generation
* **Score-entropy–based objective** for training
* **Training on Shakespeare’s text**
* **Discrete Tweedie Sampler** method for efficient inference

## Motivation

Diffusion models revolutionized image and video generation by inverting the noising process.
This project investigates how the same principle extends to discrete symbol sequences, where "noise" means flipping tokens, and "denoising" means learning to recover meaningful text.

By uniting Karpathy’s minimal GPT implementation with recent research on discrete score-matching, this notebook aims to serve as both an educational guide and a research starting point for diffusion-based language modelling.

## Acknowledgement

* A. Lou et al., *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*, [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)
* A. Lou's [Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) on GitHub.
* A. Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) on GitHub.

## Citation

If you find this notebook useful, please cite or link back to this repository.

```text
@misc{annotated_discrete_diffusion_2025,
  author = {Ashwani Kumar},
  title  = {The Annotated Discrete Diffusion Models},
  year   = {2025},
  howpublished = {\url{https://github.com/ash80/diffusion-gpt}}
}
```
