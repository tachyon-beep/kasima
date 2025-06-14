# Changelog

## [Unreleased]
### Added
- CIFAR data module with deterministic splits
- Seeded ResNet-18 backbone and AMP options
- Merkle-based germination log with verifier

## [0.2.1] - 2025-06-16
- Centralised per-seed lock creation in `SeedManager.register_seed`.
- Added `compute_health_signal` hook for `SentinelSeed`.
