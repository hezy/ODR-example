# ODR Fit with Uncertainties

This Python script performs Orthogonal Distance Regression (ODR) on data with uncertainties in both the x and y variables. It reads data from a CSV file, performs the ODR analysis, and creates plots for the data, fit, residuals, and parameter correlation ellipses.

## Features

- Reads data from a CSV file with columns for x, y, and their uncertainties (dx and dy)
- Performs ODR analysis using the `scipy.odr` module
- Calculates chi-square, reduced chi-square, and p-value for the fit
- Creates a figure with subplots for:
  - Data points with error bars and the best-fit line
  - Residuals plot with error bars
  - Parameter correlation ellipses (2D confidence regions) for the slope and intercept
- Prints the regression results, including the best-fit parameters, their uncertainties, correlation coefficient, chi-square, reduced chi-square, and p-value

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

For an interactive experience with live visualizations and the ability to modify parameters:

1. Run the marimo notebook:
```
uv run --with marimo marimo run odr-fit-marimo.py
```

2. Your web browser will automatically open to the interactive notebook interface where you can:
   - Upload and analyze your own CSV files
   - Modify fitting parameters interactively
   - View live updates of plots and statistics
   - Export results and figures

### Output Files

The command line version will save the regression results in your working directory:

`fit_results.txt`
   - Contains complete regression analysis results
   - Includes parameter correlations and statistical measures
   - Shows full covariance matrix in scientific notation

Plot files:
   - `fit_plot.png`: Data and fit line, with automatic style adjustment
   - `residuals_plot.png`: Includes propagated uncertainties from both x and y
   - `correlation_ellipses.png`: Parameter correlation ellipses at 1σ, 2σ, and 3σ confidence levels

## Visualization Features
- Automatic smart plotting: The tool dynamically chooses between showing point markers or error bars only, based on the relative size of uncertainties
  - Shows error bars only when uncertainties are significant (>1% of data range)
  - Shows points with error bars when uncertainties are smaller

## Example

An example CSV file (`data.csv`) is provided with the script that produced it (`generate-fake-data.py`). The data represents measurements with uncertainties in both x and y variables.

Running the script with the provided example data will produce the following output:

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

![fit plot](https://github.com/hezy/ODR-example/blob/main/fit_plot.png?raw=true)
![residuals plot](https://github.com/hezy/ODR-example/blob/main/residuals_plot.png?raw=true)
![correlation ellipses](https://github.com/hezy/ODR-example/blob/main/correlation_ellipses.png?raw=true)

## Best Practices

- Verify your uncertainties before analysis
- Review the residuals plot to assess fit quality
- Use the p-value and reduced chi-square to evaluate goodness-of-fit
- Check the correlation ellipses to understand parameter interdependencies

## Troubleshooting

### Common Issues
- If your CSV file isn't being read correctly, ensure it has exactly these column names: 'x', 'dx', 'y', 'dy'
- The script requires uncertainties (dx, dy) to be positive values
- For best visualization results, ensure your uncertainties are reasonable compared to your data range

### Error Messages
The script will provide clear error messages if:
- The input file cannot be found
- The CSV is missing required columns
- The data contains invalid values

## Similar Tools and Alternatives

Here are some free alternatives for fitting data with uncertainties:

### Graphical Tools
- [EddingtonGUI](https://github.com/EddLabs/eddington-gui): User-friendly GUI for curve fitting with uncertainties
  - A user-friendly graphical interface
  - Similar ODR fitting capabilities
  - Interactive plot manipulation
  - Export options for results and figures
- [Fityk](https://fityk.nieto.pl/): Versatile curve fitting tool that supports:
  - Both x and y error bars in fitting
  - Various weighting schemes
  - Linear and non-linear fitting
  - Both GUI and command-line interface

