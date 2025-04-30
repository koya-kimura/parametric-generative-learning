# parametric-generative-learning

A research-driven exploration of how artistic parameter spaces in generative art can be learned, reconstructed, and extended using machine learning.

## Overview

This project investigates the relationship between parametric control in generative art and the ability of machine learning models to learn and regenerate aesthetically meaningful outputs. The goal is to build a system that can:

- Learn from a set of generative artworks and their underlying parameters.
- Predict or regenerate parameter settings associated with "good" outputs.
- Explore the structure of latent or parametric space from an aesthetic perspective.

This work is inspired by practices in creative coding (e.g., p5.js) and seeks to connect subjective creation with formal generative models such as CGANs, VAEs, or diffusion-based techniques.

## Motivation

In creative coding, artists often tweak parameters intuitively to produce visually compelling results. This project asks:

> Can a model learn to imitate or assist those intuitive choices?

Rather than relying on large-scale datasets, this project uses self-created data and subjective evaluations as the primary source of training and supervision.

## Roadmap

- [x] Literature and project review
- [ ] Generate parameter-image datasets from p5.js sketches
- [ ] Add subjective ratings to selected outputs
- [ ] Train conditional generative models (CGAN / VAE / diffusion)
- [ ] Visualize learned structures and assess generation quality

## Related Projects

- [GenP5 & P52Style](https://github.com/KolvacS-W/GenP5-P52Style)
- [DreamBooth (Stable Diffusion)](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)
- [SinGAN](https://github.com/tamarott/SinGAN)
- [DI-PCG](https://github.com/thuzhaowang/DI-PCG)

## Author

This repository is maintained by [@koya-kimura](https://github.com/koya-kimura) as part of a master's research project in generative art and machine learning.

## License

TBD