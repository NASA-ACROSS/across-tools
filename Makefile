# local
ENV ?= local
is_local := $(filter local,$(ENV))

# define directories
VENV_DIR = .venv
VENV_BIN = $(VENV_DIR)/bin
UV_INSTALL_SCRIPT = https://astral.sh/uv/install.sh

# Detect installed tools
IS_UV_INSTALLED := $(shell command -v uv || echo "")
IS_PIP_COMPILE_INSTALLED := $(shell command -v $(VENV_BIN)/pip-compile || echo "")


# The `ask` macro displays a prompt to the user, processes their input,
# and conditionally executes a command based on the response.
#
# Args:
#   1: The message to display to the user (e.g., a question).
#   2: The command to execute if the user responds with 'y' or 'Y'.
#   3: The message to display and exit if the user responds with 'n' or 'N'.
#
# Example usage:
#   $(call ask, "Installing dependency",echo "Installing dep..." && install,"Installation aborted.")
#
# Example output:
# > Installing dependency, are you sure? (y/n)
# > y
# > Installing dep...
#
# > Installing dependency, are you sure? (y/n)
# > n
# > Installation aborted
define ask
	echo "$(1) Are you sure? (y/n)"; \
	read choice; \
	if [ "$$choice" = "y" ] || [ "$$choice" = "Y" ]; then \
		$(2); \
	else \
		echo "$(3)"; \
		exit 0; \
	fi
endef

# Tasks
.PHONY: list_targets help init install_uv install install_hooks lock venv_dir venv install_deps test lint types clean

list_targets: ### Internal command used for getting a list of commands for .PHONY
	@awk '/^[a-zA-Z_\-]+:/ {sub(/:/, ""); printf "%s ", $$1} END {print ""}' $(MAKEFILE_LIST)

help:
	@echo "Usage:\n    make \033[36m<target>\033[0m"
	@awk ' \
	BEGIN { \
		group = ""; \
	} \
	/^[#] Group:/ { \
		group = substr($$0, index($$0,$$3)); \
		print "\n" group " Commands:"; \
	} \
	/^[a-zA-Z_-]+:.*## / && !/###/ { \
		target = $$1; \
		description = substr($$0, index($$0, "##") + 3); \
		gsub(":", "", target); \
		printf "  \033[36m%-15s\033[0m %s\n", target, description \
	} \
	' $(MAKEFILE_LIST)


# Group: Setup
init: install ## Initialize the project, and dependencies

install_uv: ## Install 'uv' if needed (this will install it globally)
	@echo "Installing 'uv'...";
	@curl -LsSf $(UV_INSTALL_SCRIPT) | sh;

install: venv ## Install dependencies from the lockfile for the specified ENV
	@if [ -z $(IS_UV_INSTALLED) ]; then \
		$(call ask,\
			"'uv' is not installed. Proceeding will install to the global system.",\
			$(MAKE) install_deps,\
			"Cannot proceed without 'uv'. Exiting."\
		); \
	else \
		$(MAKE) install_deps; \
	fi;
	@$(MAKE) install_hooks

install_hooks: ## Install 'pre-commit' git hooks, only for local.
	@echo "Installing pre-commit git hooks...";
	@$(VENV_BIN)/pre-commit install; 

lock: venv ## Create or update the lockfile
	@echo "Updating lockfile...";
	uv lock

venv_dir:	### Create the venv dir if it DNE
	@mkdir -p $(VENV_DIR)

venv: venv_dir ### Create venv within specified env
	@if [ ! -d $(VENV_BIN) ]; then \
		echo "Creating virtual environment."; \
		uv venv $(VENV_DIR) --python '>=3.12,<3.13'; \
	fi;

install_deps: ### Install dependencies
	@uv sync --locked;
	@echo "Installed dependencies";


# Group: Testing
run_hooks: ## Run all pre-commit hooks against all files
	@uv run pre-commit run --all-files --verbose;
	
test: ## Run automated tests
	@$(VENV_BIN)/pytest --cov=across/tools tests/**;

test_no_cov: ## Run automated tests without coverage
	@$(VENV_BIN)/pytest tests/**;

lint: ## Run linting
	@$(VENV_BIN)/pre-commit run --all-files;

types: ## Run type checks
	@$(VENV_BIN)/mypy;


# Group: Cleaning
clean: ## Clean virtual env
	@rm -rf $(VENV_DIR)
	@echo "Cleaned up environment."