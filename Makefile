# ─────────────────────────────────────────────────────────────────────────────
# call me maybe — Makefile
# ─────────────────────────────────────────────────────────────────────────────


## install: Install project dependencies using uv
install:
	uv sync

setup_goinfre:
	mkdir -p /home/$(USER)/goinfre/call_me_brouane/uv-cache
	mkdir -p /home/$(USER)/goinfre/call_me_brouane/huggingface
	grep -qxF 'export UV_CACHE_DIR="/home/$$USER/goinfre/call_me_brouane/uv-cache"' ~/.zshrc || echo 'export UV_CACHE_DIR="/home/$$USER/goinfre/call_me_brouane/uv-cache"' >> ~/.zshrc
	grep -qxF 'export HF_HOME="/home/$$USER/goinfre/call_me_brouane/huggingface"' ~/.zshrc || echo 'export HF_HOME="/home/$$USER/goinfre/call_me_brouane/huggingface"' >> ~/.zshrc
	grep -qxF 'export UV_PROJECT_ENVIRONMENT="/home/$$USER/goinfre/call_me_brouane/.venv"' ~/.zshrc || echo 'export UV_PROJECT_ENVIRONMENT="/home/$$USER/goinfre/call_me_brouane/.venv"' >> ~/.zshrc
	@echo ""
	@echo "Run the following commands:"
	@echo "  zsh"
	@echo "  source ~/.zshrc"
	@echo "  fish"

## run: Execute the main script
run:
	uv run python -m src

## debug: Run the main script under Python's built-in debugger (pdb)
debug:
	uv run python -m pdb -m src

## clean: Remove caches and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache
	rm -rf src/.mypy_cache
	rm -rf llm_sdk/.mypy_cache
	rm -rf .ruff_cache

fclean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf src/.mypy_cache
	rm -rf llm_sdk/.mypy_cache
	rm -rf .ruff_cache
	rm -rf /home/$(USER)/goinfre/call_me_brouane

re: fclean install

## lint: Run flake8 and mypy with standard flags
lint:
	flake8 src/
	mypy src/ \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

## lint-strict: Run flake8 and mypy with strict flags (optional)
lint-strict:
	flake8 src/
	mypy src/ --strict

.PHONY: install run debug clean lint lint-strict fclean setup_goinfre re