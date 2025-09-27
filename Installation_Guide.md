# Installation Guide

This guide helps you install and run the ODR analysis tools. Choose your user type:

## Quick Navigation
- **[📓 Notebook Users](#notebook-users)** - Just want to use the interactive notebook to perform ODR analysis
- **[👩‍💻 Developers](#developers)** - Want to modify/contribute to the code
- **[🖥️ Command Line Users](#command-line-users)** - Want to use the CLI script

---

## Notebook Users

**Goal**: Run the interactive marimo notebook to perform ODR analysis

### Quick Start (Any OS)
1. Install uv package manager:
   - **Windows**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - **Linux/macOS**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Clone the repository:
   ```bash
   git clone https://github.com/hezy/ODR-example.git
   cd ODR-example
   ```
3. Run the notebook:
   ```bash
   uv run --with marimo marimo run odr-fit-marimo.py
   ```
4. Open your browser to the displayed URL (usually http://localhost:2718)

That's it! The notebook will automatically install all dependencies and provide an interactive interface for performing ODR analysis on your data.

---

## Developers

**Goal**: Modify the notebook code, contribute changes, or develop new features

### Development Setup

1. **Install prerequisites** (same as notebook users above)
2. **Clone and setup development environment**:
   ```bash
   git clone https://github.com/hezy/ODR-example.git
   cd ODR-example
   uv sync --extra dev  # Install dev dependencies (type stubs, linting tools)
   ```
3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

### Development Tools

This project includes several development tools configured in `pyproject.toml`:

- **Ruff**: Code linting and formatting
  ```bash
  uv run ruff check    # Check for issues
  uv run ruff format   # Format code
  ```
- **Pyright**: Type checking
  ```bash
  uv run pyright       # Type check all files
  ```
- **Development dependencies**: pandas-stubs, types-pytz for better type hints

### Development Workflow

1. **Edit the marimo notebook**:
   ```bash
   uv run marimo edit odr-fit-marimo.py
   ```
   This opens the notebook in edit mode where you can modify cells, add new functionality, etc.

2. **Run quality checks**:
   ```bash
   uv run ruff check && uv run ruff format && uv run pyright
   ```

3. **Test your changes**:
   ```bash
   uv run marimo run odr-fit-marimo.py  # Test notebook
   uv run odr-fit.py data.csv           # Test CLI script
   ```

### Project Structure for Developers
- `odr-fit-marimo.py`: Main marimo notebook with interactive cells
- `odr-fit.py`: Original CLI script (shares core functions)
- `pyproject.toml`: Project configuration, dependencies, tool settings
- `uv.lock`: Locked dependency versions for reproducibility

---

## Command Line Users

**Goal**: Use the CLI script for batch processing of your data

### Quick Start
1. Install uv (same as notebook users above)
2. Clone the repository:
   ```bash
   git clone https://github.com/hezy/ODR-example.git
   cd ODR-example
   ```
3. Run with example data:
   ```bash
   uv run odr-fit.py data.csv
   ```
4. Or with your own CSV file (must have columns: x, dx, y, dy):
   ```bash
   uv run odr-fit.py your_data.csv
   ```

### Output Files
- `fit_results.txt`: Statistical analysis results
- `fit_plot.png`: Data and fit visualization
- `residuals_plot.png`: Residuals analysis
- `correlation_ellipses.png`: Parameter correlation ellipses

---

## Detailed OS-Specific Instructions

**Note**: Most users can use the quick start sections above. These detailed instructions are for users who need step-by-step guidance or encounter issues.

### Prerequisites Installation

#### Windows
1. **Install uv**:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   Then restart PowerShell.

2. **Install Git** (if not already installed):
   - Download from: https://git-scm.com/download/windows
   - Use default installation options

#### Linux
1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc  # or restart terminal
   ```

2. **Install Git** (if not already installed):
   - Ubuntu/Debian: `sudo apt update && sudo apt install git`
   - Fedora/RHEL: `sudo dnf install git`
   - Arch: `sudo pacman -S git`

#### macOS
1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc  # or restart terminal
   ```

2. **Install Git** (if not already installed):
   ```bash
   xcode-select --install
   # or via Homebrew: brew install git
   ```

---

## Troubleshooting

### Common Issues

**Data Format Problems**:
- CSV must have columns named exactly: `x`, `dx`, `y`, `dy`
- Uncertainties (`dx`, `dy`) must be positive values
- Remove any NaN/missing values from your data

**Installation Issues**:
- `'uv' is not recognized`: Restart your terminal/PowerShell and try again
- `'git' is not recognized`: Install Git using instructions above
- Dependencies fail to install: Try running `uv sync` first

**Marimo Notebook Issues**:
- Browser doesn't open automatically: Navigate to the URL shown in terminal (usually http://localhost:2718)
- Can't edit notebook: Use `uv run marimo edit odr-fit-marimo.py` instead of `marimo run`
- Port conflicts: marimo will automatically try different ports if 2718 is busy

### Verification
Test your installation:
1. **CLI version**: `uv run odr-fit.py data.csv` should create 4 output files
2. **Notebook version**: `uv run --with marimo marimo run odr-fit-marimo.py` should open browser
3. **Results comparison**: Both versions should produce identical statistical results

### Getting Help
- Check the example data works first: `uv run odr-fit.py data.csv`
- Compare your CSV format to the provided `data.csv`
- Error messages are designed to be educational - read them carefully!
