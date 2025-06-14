# kasima

Morphogenetic Architectures experiment demonstrating adaptive network growth.

This project implements a minimal "morphogenetic engine" inspired by the
Morphogenetic Architectures framework. The accompanying experiment script trains
a small network on the classic two spirals classification task and explores a
few hyperparameters to maximise validation accuracy. The library now ships with
helpers for CIFAR-10/100 experiments and a ResNet backbone.

## Dependencies

The code targets **Python 3.12** and relies on the following packages (see
`requirements.txt` for exact versions):

- `numpy`
- `scikit-learn`
- `torch`
- `torchvision`

For development and testing you may also install the tools listed in
`requirements-dev.txt` (`pytest`, `black`, `ruff`).

## Running the experiment

Install the requirements and install the package in editable mode, then run the
experiment module from the project root:

```bash
pip install -r requirements.txt
pip install -e .
python -m scripts.run_experiment --device cpu --amp
```

For CIFAR-10/100 runs, pass a Hydra config name:

```bash
python -m scripts.run_experiment --config-name dataset=cifar10,model=resnet18
```

The script prints the randomly selected hyperparameters, training progress, and
any germination events. A successful run should eventually reach high validation
accuracy (>0.9). Example tail output:

```text
Epoch 296/296 - loss: 0.0000, acc: 1.0000
Best validation accuracy: 1.0
Germination log:
{'seed_id': 'seed2', 'success': True, 'timestamp': 1749911919.5063906}
{'seed_id': 'seed1', 'success': True, 'timestamp': 1749911926.4253361}
```

To verify a germination log, use the ``kaslog`` tool:

```bash
python -m kaslog verify germination.jsonl
```
