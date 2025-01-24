# Specify the names of all executables to make.
PROG=update install style_check
.PHONY: ${PROG}

update:
	pip install --upgrade pip wheel
	pip install --upgrade -r requirements.txt

install:
	pip install --upgrade pip wheel
	pip install -r requirements.txt
	git config core.hooksPath .github/hooks

style_check:
	black . --config pyproject.toml
	isort . --gitignore --settings-path pyproject.toml
