# $\text{CO}_2$ Modelling with Hidden Markov Models

This project is a special course at the Technical University of Denmark (DTU) exploring how to model $\text{CO}_2$ data with Hidden Markov Models. It was conducted in the Spring Semester 2026 (5 ECTS) and leads up to a Bachelor Project. The focus is on designing flexible HMM software, exploring and interpreting HMM states, and testing different HMM variants (ordinary, AR, second-order) on $\text{CO}_2$ time series data.

Package manager: `uv`.

## Repository layout

- **`src/`** — The HMM library. Contains the base classes (emission, transition, HMM), optimisation utilities, data loading, and the versioned APIs under `src/api/v1`–`v4` that compose emissions and transitions into concrete models.
- **`tests/`** — Unit and integration tests for the library. Only `tests/v4` is current.
- **`drivers/`** — Runnable entry points that use `src` to fit models, generate plots, and produce the result tables/figures consumed by the report. See `drivers/models/` for one driver per model variant.
- **`report/`** — LaTeX source for the written report (`main.tex`, `sections/`, `preamble/`, `references.bib`) along with build artefacts and the compiled `main.pdf`.

## API versioning — only v4 is active

The `src/api/` directory contains four iterations of the modelling API (`v1`, `v2`, `v3`, `v4`). **Only `v4` should be treated as the working, supported API.** All earlier versions (`v1`, `v2`, `v3`) and anything under `src/deprecated/` are kept for historical reference only and should not be used or extended. New drivers, tests, and report results target `src/api/v4` exclusively.

## AI disclosure

1. Copilot chat completion has been used.
2. No agents have written code.
3. Agents have been used to debug JAX modules (sometimes).
4. Claude has been used to find sources and explain concepts.
5. Claude Code has been used to generate documentation about the code.
