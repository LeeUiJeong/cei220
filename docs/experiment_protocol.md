# Experiment Protocol

## Design

Four dataset-model combinations receive the same 55-run protocol:

1. CIFAR-10 + SmallCNN
2. CIFAR-10 + ResNet18-small
3. Fashion-MNIST + SmallCNN
4. Fashion-MNIST + ResNet18-small

Per combination:

- baseline: 10 runs
- CEI-lite: 10 runs
- CEI-strict: 10 runs
- perturbed (log-only event): 10 runs
- external-seed generalization: 3 seeds × 3 runs = 9 runs
- loader-seed perturbation: 3 runs
- model-seed perturbation: 3 runs

Total: 55 × 4 = 220 runs.

## Shared execution engine

All combinations use the same:

- HKDF derivation implementation
- hierarchical namespace scheme
- policy profiles
- training/evaluation loop
- model-state hash algorithm
- artifact schema
- environment fingerprint
- analysis and verification pipeline

Dataset and model differences are configuration parameters rather than separate execution implementations.
