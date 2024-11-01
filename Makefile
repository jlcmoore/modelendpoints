.PHONY: init init-conda test

init:
	python3.11 -m venv env-modelendpoints
	env-modelendpoints/bin/pip install --editable .

test:
	RUN_QUERY_TESTS=True \
	env-modelendpoints/bin/pytest tests -W error

init-conda:
	conda env create --file environment.yml
	conda activate env-modelendpoints && \
	pip install --editable .
