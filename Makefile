.PHONY: help build check publish publish-test clean

help:
	@echo "make build         build sdist + wheel into dist/ (no upload)"
	@echo "make publish       build and upload to PyPI using pypi_api from .env"
	@echo "make publish-test  build and upload to TestPyPI using testpypi_api from .env"
	@echo "make clean         remove build artifacts"

build check:
	@scripts/publish.sh --dry-run

publish:
	@scripts/publish.sh

publish-test:
	@scripts/publish.sh --test

clean:
	rm -rf dist build src/*.egg-info
