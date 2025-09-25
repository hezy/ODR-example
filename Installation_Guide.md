# Installation Guide

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

### Option A: Command Line Version
1. Make sure you're in the project directory (from Step 3)
2. uv will automatically manage Python and install dependencies when you run the script:
   ```bash
   uv run odr-fit.py data.csv
   ```

### Option B: Interactive Marimo Notebook Version
1. For an interactive experience with live visualizations, run the marimo notebook:
   ```bash
   uv run --with marimo marimo run odr-fit-marimo.py
   ```
2. Your web browser will automatically open to show the interactive notebook
3. In the notebook interface you can:
   - Upload and analyze your own CSV files
   - Modify parameters and see results update in real-time
   - View interactive plots and statistics
   - Export results and figures

That's it! uv will automatically:
- Install the correct Python version if needed
- Create a virtual environment
- Install all required packages from requirements.txt
- Install marimo (for the notebook version)
- Run your chosen script

---

## Troubleshooting

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
