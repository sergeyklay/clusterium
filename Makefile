include defaults.mk

.PHONY: install
install:
	@echo $(CS)Installing $(PKG_NAME) package: $(PKG_DIR)$(CE)
	poetry install
	@echo

.PHONY: test
test:
	@echo $(CS)Running tests for package: $(PKG_NAME)$(CE)
	$(VENV_BIN)/coverage erase
	$(VENV_BIN)/coverage run -m pytest -m "not integration" $(PYTEST_FLAGS) ./$(PKG_NAME) ./tests
	$(VENV_BIN)/coverage run -m pytest -m integration $(PYTEST_FLAGS) ./$(PKG_NAME) ./tests
	@echo

.PHONY: ccov
ccov:
	@echo $(CS)Combine coverage reports for package: $(PKG_NAME)$(CE)
	@mkdir -p coverage/html coverage/xml coverage/lcov
	$(VENV_BIN)/coverage combine || true
	$(VENV_BIN)/coverage report
	$(VENV_BIN)/coverage html -d coverage/html
	$(VENV_BIN)/coverage xml -o coverage/xml/coverage.xml

.PHONY: format
format:
	@echo $(CS)Formatting code for package: $(PKG_NAME)$(CE)
	$(VENV_BIN)/isort --profile black --python-version auto ./
	$(VENV_BIN)/black . ./$(PKG_NAME) ./tests
	@echo

.PHONY: format-check
format-check:
	@echo $(CS)Checking formatting for package: $(PKG_NAME)$(CE)
	$(VENV_BIN)/isort --check-only --profile black --python-version auto --diff ./
	$(VENV_BIN)/black --check . ./$(PKG_NAME) ./tests
	@echo

.PHONY: lint
lint:
	@echo $(CS)Running linters for package: $(PKG_NAME)$(CE)
	$(VENV_BIN)/flake8 $(FLAKE8_FLAGS) ./
	$(VENV_BIN)/pylint $(FLAKE8_FLAGS) ./$(PKG_NAME)
	@echo

.PHONY: clean
clean:
	@echo $(CS)Remove build and tests artefacts and directories$(CE)
	find ./ -name '__pycache__' -delete -o -name '*.pyc' -delete
	$(RM) -r ./.pytest_cache
	$(RM) ./coverage
	@echo

.PHONY: docs
docs: CONTRIBUTING.rst README.rst
	@echo $(CS)Building documentation for package: $(PKG_NAME)$(CE)
	$(VENV_PYTHON) -m doctest CONTRIBUTING.rst README.rst
	$(VENV_PYTHON) -m sphinx --jobs auto --builder html --nitpicky --show-traceback --fail-on-warning --doctree-dir docs/build/doctrees docs/source docs/build/html

$(VENV_PYTHON): $(VENV_ROOT)
	@echo

$(VENV_ROOT):
	@echo $(CS)Creating a Python environment $(VENV_ROOT)$(CE)
	$(VIRTUALENV) --prompt $(VENV_NAME) $(VENV_ROOT)
	@echo
	@echo Done.
	@echo
	@echo To active it manually, run:
	@echo
	@echo "    source $(VENV_BIN)/activate"
	@echo
	@echo See https://docs.python.org/3/library/venv.html for more.
	@echo
	$(call mk-venv-link)


.PHONY: help
help:
	@echo "$(TITLE)$(PKG_NAME) Build System$(RESET)"
	@echo
	@echo "$(SECTION)Available targets:$(RESET)"
	@echo
	@echo "  $(TARGET)help$(RESET)         $(DESCRIPTION)Show this help message$(RESET)"
	@echo "  $(TARGET)install$(RESET)      $(DESCRIPTION)Install $(PKG_NAME) and its dependencies$(RESET)"
	@echo "  $(TARGET)test$(RESET)         $(DESCRIPTION)Run tests for $(PKG_NAME)$(RESET)"
	@echo "  $(TARGET)ccov$(RESET)         $(DESCRIPTION)Generate combined coverage reports$(RESET)"
	@echo "  $(TARGET)format$(RESET)       $(DESCRIPTION)Format code in $(PKG_NAME)$(RESET)"
	@echo "  $(TARGET)format-check$(RESET) $(DESCRIPTION)Check code formatting in $(PKG_NAME)$(RESET)"
	@echo "  $(TARGET)lint$(RESET)         $(DESCRIPTION)Run linters in $(PKG_NAME)$(RESET)"
	@echo "  $(TARGET)docs$(RESET)         $(DESCRIPTION)Build documentation for $(PKG_NAME)$(RESET)"
	@echo "  $(TARGET)clean$(RESET)        $(DESCRIPTION)Remove build and tests artefacts and directories$(RESET)"
	@echo
	@echo '$(SECTION)Virtualenv:$(RESET)'
	@echo
	@echo "  Python:       $(VENV_PYTHON)"
	@echo "  pip:          $(VENV_PIP)"
	@echo "  Virtualenv:   $(if $(VENV_NAME),$(VENV_NAME),N/A)"
	@echo
	@echo "$(SECTION)Flags:$(RESET)"
	@echo
	@echo "  FLAKE8_FLAGS: $(FLAKE8_FLAGS)"
	@echo "  PYTEST_FLAGS: $(PYTEST_FLAGS)"
	@echo "  PYLINT_FLAGS: $(PYLINT_FLAGS)"
	@echo
	@echo "$(SECTION)Environment variables:$(RESET)"
	@echo
	@echo "  PYTHON:                $(if $(PYTHON),$(PYTHON),N/A)"
	@echo "  PYENV_ROOT:            $(if $(PYENV_ROOT),$(PYENV_ROOT),N/A)"
	@echo "  PYENV_VIRTUALENV_INIT: $(if $(PYENV_VIRTUALENV_INIT),$(PYENV_VIRTUALENV_INIT),N/A)"
	@echo "  PYENV_SHELL:           $(if $(PYENV_SHELL),$(PYENV_SHELL),N/A)"
	@echo "  WORKON_HOME:           $(if $(WORKON_HOME),$(WORKON_HOME),N/A)"
	@echo "  VIRTUAL_ENV:           $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV),N/A)"
	@echo "  SHELL:                 $(if $(SHELL),$(SHELL),N/A)"
	@echo "  TERM:                  $(if $(TERM),$(TERM),N/A)"
	@echo
