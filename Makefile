PYTHON ?= python

.PHONY: install lint test train evaluate ui clean all

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m flake8 src tests --max-line-length 100

test:
	$(PYTHON) -m pytest -v

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage

all: install lint test train evaluate
