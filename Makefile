# Makefile for covariance_calculators

VENV_NAME := .venv
CODE_DIRS := tests *.py

.PHONY: all install clean realclean test testclean lint blackcheck black submodules

all: lint install

install: $(VENV_NAME)
	@echo "\nTo start, please run\nsource setup.sh\n"

$(VENV_NAME):
	bash setup.sh

clean: testclean
	find . -type f -name '*.py[co]' -exec rm -fv {} +
	find . -type d -name __pycache__  -exec rm -rfv {} +

distclean: clean
	find . -maxdepth 1 -type d -name $(VENV_NAME) -exec rm -rfv {} +

realclean: distclean

test:
	cd tests && pytest && cd ..

testclean:
	find tests -type f -name '*.py[co]' -exec rm -fv {} +
	find tests -type d -name __pycache__  -exec rm -rfv {} +
	find tests -type d -name .pytest_cache -exec rm -rfv {} +
	find tests -type f -name '*.csv' -exec rm -fv {} +
	find tests -type f -name '*.png' -exec rm -fv {} +
	find tests -type f -name '*.pdf' -exec rm -fv {} +

lint:
	flake8 $(CODE_DIRS) --count --select=E9,E711,E712,F4,F7,F63,F82,F841,W605 --show-source --statistics

blackcheck:
	find $(CODE_DIRS) -name \*\.py | xargs black --check

black:
	find $(CODE_DIRS) -name \*\.py | xargs black

