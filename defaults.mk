# Common variables and utilities for all makefiles

ROOT_DIR := $(shell git rev-parse --show-toplevel)
SHELL    := $(shell which bash)

.DEFAULT_GOAL := help

ifndef VERBOSE
MAKEFLAGS += --no-print-directory
endif

FLAKE8_FLAGS ?=
PYLINT_FLAGS ?=

# Terminal color support detection and configuration
# Check if we're in a CI environment (GitHub Actions)
CI_ENV := $(if $(GITHUB_ACTIONS),1,$(if $(CI),1,0))

# Only use tput if we have a valid TERM and not in CI
COLORIZE := $(shell if [ -n "$$TERM" ] && [ "$$TERM" != "dumb" ] && [ "$(CI_ENV)" != "1" ]; then command -v tput >/dev/null 2>&1 && echo 1 || echo 0; else echo 0; fi)

ifeq ($(COLORIZE),1)
	# Regular colors
	BLACK        := $(shell tput setaf 0)
	RED          := $(shell tput setaf 1)
	GREEN        := $(shell tput setaf 2)
	YELLOW       := $(shell tput setaf 3)
	BLUE         := $(shell tput setaf 4)
	MAGENTA      := $(shell tput setaf 5)
	CYAN         := $(shell tput setaf 6)
	WHITE        := $(shell tput setaf 7)

	# Bold colors
	BOLD         := $(shell tput bold)
	BOLD_GREEN   := $(BOLD)$(GREEN)
	BOLD_YELLOW  := $(BOLD)$(YELLOW)
	BOLD_BLUE    := $(BOLD)$(BLUE)
	BOLD_MAGENTA := $(BOLD)$(MAGENTA)

	# Special
	RESET        := $(shell tput sgr0)

	# Composite styles for different types of content
	TITLE        := $(BOLD_BLUE)
	SECTION      := $(BOLD_MAGENTA)
	TARGET       := $(BOLD_GREEN)
	DESCRIPTION  := $(WHITE)
	EXAMPLE      := $(BOLD_YELLOW)
else
	# No color support
	BLACK := YELLOW := RED := GREEN := BLUE := MAGENTA := CYAN := WHITE := \
	BOLD := BOLD_GREEN := BOLD_YELLOW := BOLD_BLUE := BOLD_MAGENTA := \
	RESET := TITLE := SECTION := TARGET := DESCRIPTION := EXAMPLE :=
endif

PKG_NAME := "clusx"

# Color output configuration
ifneq ($(TERM),)
	CS = "${GREEN}~~~ "
	CE = " ~~~${RESET}"

	PYTEST_FLAGS ?= --color=yes
else
	CS = "~~~ "
	CE = " ~~~"

	PYTEST_FLAGS ?=
endif

# Python environment setup
HAVE_PYENV := $(shell sh -c "command -v pyenv")

ifneq ($(VIRTUAL_ENV),)
	VENV_ROOT = $(VIRTUAL_ENV)
	VENV_NAME = $(shell basename $(VIRTUAL_ENV))
else
ifneq ($(HAVE_PYENV),)
	VENV_NAME = $(shell pyenv version-name)
	VENV_ROOT = $(shell pyenv root)/versions/$(VENV_NAME)
else
	VENV_ROOT = .venv
	VENV_NAME = engagement-portal
endif
endif

# OS-specific configurations
ifeq ($(OS),Windows_NT)
	PYTHON  ?= python.exe
	VIRTUALENV ?= virtualenv.exe
	VENV_BIN = $(VENV_ROOT)/Scripts
else
	PYTHON  ?= python3
	VIRTUALENV ?= $(PYTHON) -m venv
	VENV_BIN = $(VENV_ROOT)/bin
endif

VENV_PYTHON = $(VENV_BIN)/python
VENV_PIP    = $(VENV_PYTHON) -m pip

export PATH := $(VENV_BIN):$(PATH)

# Python availability check
ifndef PYTHON
$(error "Python is not available please install Python")
else
ifneq ($(OS),Windows_NT)
HAVE_PYTHON := $(shell sh -c "command -v $(PYTHON)")
ifndef HAVE_PYTHON
$(error "Python is not available. Please install Python.")
endif
endif
endif

# Common utility functions
define mk-venv-link
	@if [ -n "$(WORKON_HOME)" ] ; then \
		echo $(ROOT_DIR) > $(VENV_ROOT)/.project; \
		if [ ! -d $(WORKON_HOME)/$(VENV_NAME) -a ! -L $(WORKON_HOME)/$(VENV_NAME) ]; \
		then \
			ln -s $(ROOT_DIR)/$(VENV_ROOT) $(WORKON_HOME)/$(VENV_NAME); \
			echo ; \
			echo Since you use virtualenvwrapper, we created a symlink; \
			echo "so you can also use \"workon $(VENV_NAME)\" to activate the venv."; \
			echo ; \
		fi; \
	fi
endef
