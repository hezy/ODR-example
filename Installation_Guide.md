# Installation Guide

This guide helps you install and run both versions of the ODR analysis tools:
- **Command-line version** (`odr-fit.py`) for batch processing
- **Interactive marimo notebook** (`odr-fit-marimo.py`) for educational exploration

Choose your operating system for specific installation instructions:

- [Windows](#windows)
- [Linux](#linux)
- [macOS](#macos)

---

## Windows

### Step 1: Install uv
1. Open PowerShell (press Windows key + X, then select "PowerShell")
2. Run the installation command:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. Restart PowerShell after installation

### Step 2: Install Git
1. Download Git from: https://git-scm.com/download/windows
2. Run the installer
3. Keep all default options during installation - just keep clicking "Next"
4. Click "Install" when you reach that screen
5. Click "Finish" when installation is complete

### Step 3: Download This Script
1. Open Command Prompt or PowerShell:
   - Press Windows key + R, type `cmd` and press Enter, OR
   - Press Windows key + X, then select "PowerShell"
2. Navigate to where you want to download the script:
   ```cmd
   cd Desktop
   ```
3. Clone this repository:
   ```cmd
   git clone https://github.com/hezy/ODR-example.git
   ```
4. Enter the project directory:
   ```cmd
   cd ODR-example
   ```

---

## Linux

### Step 1: Install uv
1. Open Terminal (Ctrl+Alt+T)
2. Run the installation command:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Restart your terminal or run: `source ~/.bashrc`

### Step 2: Install Git
Git is usually pre-installed on most Linux distributions. If not:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install git
```

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install git
```

**Arch Linux:**
```bash
sudo pacman -S git
```

### Step 3: Download This Script
1. Open Terminal
2. Navigate to where you want to download the script:
   ```bash
   cd ~/Desktop
   ```
3. Clone this repository:
   ```bash
   git clone https://github.com/hezy/ODR-example.git
   ```
4. Enter the project directory:
   ```bash
   cd ODR-example
   ```

---

## macOS

### Step 1: Install uv
1. Open Terminal (Cmd+Space, type "Terminal")
2. Run the installation command:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Restart Terminal or run: `source ~/.bashrc`

### Step 2: Install Git
Git is usually included with Xcode Command Line Tools. To install:

```bash
xcode-select --install
```

Alternatively, you can install via Homebrew (if you have it):
```bash
brew install git
```

### Step 3: Download This Script
1. Open Terminal
2. Navigate to where you want to download the script:
   ```bash
   cd ~/Desktop
   ```
3. Clone this repository:
   ```bash
   git clone https://github.com/hezy/ODR-example.git
   ```
4. Enter the project directory:
   ```bash
   cd ODR-example
   ```

---

## Running the Scripts (All Operating Systems)

Both versions use the same core analysis functions and produce equivalent results, making them perfect for educational comparison.

### Option A: Command Line Version
**Best for**: Batch processing your own data files

1. Make sure you're in the project directory (from Step 3)
2. Run with the provided example data:
   ```bash
   uv run odr-fit.py data.csv
   ```
3. Or with your own CSV file (must have columns: x, dx, y, dy):
   ```bash
   uv run odr-fit.py your_data.csv
   ```
4. Results will be saved as:
   - `fit_results.txt` - Statistical analysis
   - `fit_plot.png` - Data and fit visualization
   - `residuals_plot.png` - Residuals analysis
   - `correlation_ellipses.png` - Parameter correlations

### Option B: Interactive Marimo Notebook Version
**Best for**: Learning and exploring ODR analysis interactively

1. Run the marimo notebook:
   ```bash
   uv run --with marimo marimo run odr-fit-marimo.py
   ```
2. Your web browser will automatically open to show the interactive notebook
3. In the notebook interface you can:
   - **Learn step-by-step**: Each analysis step is explained with educational content
   - **Compare with known values**: Uses synthetic data with known true parameters
   - **Upload your own data**: Same CSV format as the CLI version
   - **See real-time updates**: Modify parameters and see immediate results
   - **Understand the code**: Same core functions as the CLI version, clearly documented

### Educational Workflow
For best learning experience:
1. **Start with the marimo notebook** to understand the analysis interactively
2. **Examine the source code** of both versions to see the shared functions
3. **Use the CLI version** to process your own datasets
4. **Compare results** between both versions to verify consistency

### What uv Does Automatically
- Installs the correct Python version if needed
- Creates an isolated virtual environment
- Installs all required packages (NumPy, SciPy, matplotlib, pandas)
- Installs marimo (for the notebook version)
- Manages all dependencies automatically

---

## Troubleshooting

### General Issues (Both Versions)
- **CSV format problems**: Ensure your file has columns named exactly: 'x', 'dx', 'y', 'dy'
- **Negative uncertainties**: Both dx and dy must be positive values
- **Missing data**: Remove or interpolate any NaN/missing values

### Windows
- If you get "'uv' is not recognized": Restart PowerShell/Command Prompt and try again
- If you get "'git' is not recognized": Restart PowerShell/Command Prompt and try again
- If you get errors during uv run: Try running:
  ```cmd
  uv sync
  uv run odr-fit.py data.csv
  ```
- For marimo notebook issues: Try:
  ```cmd
  uv sync
  uv run --with marimo marimo run odr-fit-marimo.py
  ```
- **Browser doesn't open for marimo**: Manually navigate to the URL shown in the terminal

### Linux/macOS
- If you get "'uv' is not recognized": Restart your terminal or run `source ~/.bashrc`
- If you get "'git' is not recognized": Make sure git is installed (see Step 2 above)
- If you get errors during uv run: Try running:
  ```bash
  uv sync
  uv run odr-fit.py data.csv
  ```
- For marimo notebook issues: Try:
  ```bash
  uv sync
  uv run --with marimo marimo run odr-fit-marimo.py
  ```
- **Firewall blocking marimo**: Check firewall settings if browser doesn't open

### Verification Steps
To verify both versions are working correctly:
1. **Test CLI version**: Should create 4 output files when run successfully
2. **Test marimo version**: Should open browser and display interactive notebook
3. **Compare results**: Both versions should produce identical statistical results for the same data
4. **Check example data**: Run both versions with `data.csv` to verify installation

### Educational Troubleshooting
- **Understanding differences**: If results differ, check that both versions use the same input data
- **Learning from errors**: Error messages in both versions are designed to be educational
- **Code comparison**: If confused, compare the source code to see shared functions
