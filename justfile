# Run a specific script
# usage: just run experiments/my_script.py
run script:
    uv run python {{script}}

# Run Jupyter Lab
notebook:
    uv run jupyter lab

# Run lint and format
lint:
    uv run ruff check --fix
    uv run ruff format

# Clean all caches
clean:
    find . -type d \( -name "__pycache__" -o -name ".ruff_cache" \) -exec rm -rf {} +
