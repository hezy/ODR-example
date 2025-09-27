# ODR Fit with Uncertainties

This educational project provides two complementary implementations of Orthogonal Distance Regression (ODR) analysis for data with uncertainties in both x and y variables:

1. **`odr-fit.py`** - Command-line script for batch processing
2. **`odr-fit-marimo.py`** - Interactive marimo notebook for educational exploration

Both implementations share identical core analysis functions and produce equivalent statistical results.

## Features

### Core Analysis Capabilities (Identical in Both Versions)
- Reads data from CSV files with columns: x, dx, y, dy (uncertainties)
- Performs ODR analysis using the `scipy.odr` module with full uncertainty propagation
- Calculates comprehensive goodness-of-fit statistics:
  - Chi-square and reduced chi-square
  - P-value for statistical significance
  - Parameter correlation coefficients
- Creates three key visualizations:
  - Data points with error bars and fitted line
  - Residuals plot with propagated uncertainties
  - Parameter correlation ellipses (1σ, 2σ, 3σ confidence regions)

### Command-Line Version (`odr-fit.py`)
- Batch processing with command-line arguments
- Saves results to files (plots as PNG, statistics as TXT)
- Optimized for automated analysis workflows

### Interactive Notebook Version (`odr-fit-marimo.py`)
- Live, interactive exploration with marimo
- Real-time visualization updates
- Inline documentation explaining each analysis step
- Direct comparison with known true parameter values

## Dependencies

- Python 3.x
- NumPy
- Pandas
- SciPy
- Matplotlib

## Installation

Clone this repository, and install the dependencies using uv.
Detailed installation guide for beginners here:
https://github.com/hezy/ODR-example/blob/main/Installation_Guide.md

## Usage

### Command Line Version

1. Prepare a CSV file with the following columns: x, dx, y, dy
   - x: x-values of the data points
   - dx: uncertainties in the x-values
   - y: y-values of the data points
   - dy: uncertainties in the y-values

2. Run the script:
```
uv run odr-fit.py data.csv
```
(change to your file name)

### Interactive Marimo Notebook Version

For educational exploration with interactive visualizations:

1. Run the marimo notebook:
```
uv run --with marimo marimo run odr-fit-marimo.py
```

2. Your web browser will automatically open to the interactive notebook interface where you can:
   - Upload and analyze your CSV files (same format as CLI version)
   - See real-time comparison between fitted and true parameter values
   - Understand each analysis step through educational documentation
   - View the same plots and statistics as the CLI version, but inline

### Structural Similarities for Learning

Both versions are intentionally structured with:
- **Identical core functions**: `read_data()`, `perform_odr()`, `confidence_ellipse()`, etc.
- **Same analysis workflow**: Data loading → ODR fitting → Statistical analysis → Visualization
- **Consistent variable naming** and function signatures
- **Equivalent statistical calculations** and goodness-of-fit tests
- **Same plotting logic** with identical error bar handling and styling

### Output Comparison

**Command-Line Version (`odr-fit.py`) - File Output:**

`fit_results.txt` - Complete regression analysis:
   - Best-fit parameters with uncertainties
   - Covariance matrix in scientific notation
   - Pearson correlation coefficient
   - Chi-square statistics and p-value

Plot files:
   - `fit_plot.png`: Data and fit line with automatic style adjustment
   - `residuals_plot.png`: Residuals with propagated uncertainties
   - `correlation_ellipses.png`: Parameter correlation ellipses (1σ, 2σ, 3σ)

**Marimo Version (`odr-fit-marimo.py`) - Interactive Display:**
   - Same statistical results displayed inline with educational formatting
   - Same plots rendered interactively in the browser
   - Additional educational features:
     - Comparison with known true parameter values
     - Real-time parameter updates
     - Step-by-step analysis explanation

## Visualization Features
- Automatic smart plotting: The tool dynamically chooses between showing point markers or error bars only, based on the relative size of uncertainties
  - Shows error bars only when uncertainties are significant (>1% of data range)
  - Shows points with error bars when uncertainties are smaller

## Example

An example CSV file (`data.csv`) is provided with the script that produced it (`generate_fake_data.py`). The data represents synthetic measurements with uncertainties in both x and y variables, perfect for testing both implementations.

### Example Results

Running either version with the provided example data will produce equivalent statistical results. The CLI version output:

```
Regression Results:
-----------------
Slope: 2.147160 ± 0.016329
Intercept: -0.359677 ± 0.137575

Covariance matrix:
 2.369639e-04 -1.720251e-03
-1.720251e-03  1.682074e-02
Pearson's Correlation coefficient: -0.765758

Chi-square: 14.627831
Degrees of freedom: 13
Reduced chi-square: 1.125218
P-value: 0.331159
```

The marimo version displays the same statistical information interactively with additional educational context and real-time parameter comparison.

![fit plot](https://github.com/hezy/ODR-example/blob/main/fit_plot.png?raw=true)
![residuals plot](https://github.com/hezy/ODR-example/blob/main/residuals_plot.png?raw=true)
![correlation ellipses](https://github.com/hezy/ODR-example/blob/main/correlation_ellipses.png?raw=true)

## Analysis Best Practices (Both Versions)
- Verify your uncertainties before analysis
- Review the residuals plot to assess fit quality
- Use the p-value and reduced chi-square to evaluate goodness-of-fit
- Check the correlation ellipses to understand parameter interdependencies

## Troubleshooting

### Common Issues (Both Versions)
- **CSV format**: Ensure your file has exactly these column names: 'x', 'dx', 'y', 'dy'
- **Positive uncertainties**: Both dx and dy must be positive values
- **Reasonable uncertainties**: For best visualization, ensure uncertainties are reasonable compared to your data range

### Version-Specific Issues

**Command-Line Version:**
- File not found errors: Check file path and working directory
- Permission errors: Ensure write permissions for output files

**Marimo Version:**
- Browser not opening: Check firewall settings or manually navigate to displayed URL
- Interactive elements not responding: Refresh the browser page
- File upload issues: Use the same CSV format as the CLI version

### Error Messages
Both versions provide clear error messages for:
- Input file issues (missing, unreadable, wrong format)
- Missing required CSV columns
- Invalid data values (negative uncertainties, NaN values)

## Alternative tools:

- [EddingtonGUI](https://github.com/EddLabs/eddington-gui): GUI-based curve fitting with uncertainties
- [Fityk](https://fityk.nieto.pl/): Professional curve fitting with both CLI and GUI interfaces
- Standard scipy.optimize tools: For comparison with non-ODR approaches

