.PHONY: validate validate-smoke validate-full test

validate:
	python validate_artifacts.py --metadata-only

validate-smoke:
	python validate_artifacts.py --mode smoke

validate-full:
	python validate_artifacts.py --mode full

test:
	python -m unittest discover -s tests
