install:
		pip install -r requirements.txt
install-dev:
		pip install -r requirements-dev.txt
		pre-commit install
run:
		python -m src