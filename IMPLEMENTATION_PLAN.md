# Causal Chess — Implementation Plan

A Python chess engine that learns its evaluation function **during search** via temporal-difference updates on a convolutional neural network.

## Decisions

- **Score convention**: White-relative value function V(s) ∈ [0, 1] (probability of White winning)
- **Top-n breadth**: Configurable, default n=5
- **Learning rate**: Fixed 1e-4 to start
- **Search depth**: Configurable, default 4
- **Self-play**: Loop with PGN game output for progress visibility
- **Device**: Configurable (cpu/mps/cuda), test on Apple Silicon
- **Online learning**: Value function updates during search (non-deterministic by design)

## Architecture

See implementation_plan.md artifact for full details.
