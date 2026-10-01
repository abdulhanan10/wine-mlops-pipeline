# Wine MLOps Pipeline

![CI](https://github.com/abdulhanan10/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

Reproducible MLOps pipeline on `sklearn.datasets.load_wine`: Makefile automation,
MLflow tracking + model registry, and a GitHub Actions model quality gate.

## Quick start
```bash
make install   # install pinned dependencies
make lint      # flake8 (max line length 100)
make test      # pytest incl. model quality gate
make train     # 8 MLflow runs, registers WineClassifier@champion
make evaluate  # loads champion, reports test metrics
make ui        # open MLflow UI at http://127.0.0.1:5000
make clean
```
All splits and models use seed 42.

## Quality gate
Validation macro F1 >= 0.88, batch inference <= 30 ms, class indices in {0,1,2}.
