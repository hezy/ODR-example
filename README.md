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

Clone this repository, and install the dependencies.
Detailed installation guide for beginners here:
https://github.com/hezy/ODR-example/blob/main/Installation_Guide.md

## Usage

1. Prepare a CSV file with the following columns: x, dx, y, dy
   - x: x-values of the data points
   - dx: uncertainties in the x-values
   - y: y-values of the data points
   - dy: uncertainties in the y-values

2. Run the script:
   ```
   python odr-fit.py data.csv
(change to your file name)

3. The script will read the data from the CSV file, perform the ODR analysis, and save the regression results in a file `fit_results.txt`

5. Three figures will be saved as files:
* `fit_plot.png` - the data points with best-fit line
* `residuals_plots.png` - the differences between observed y's to fitted y's
* `correlation_ellipses.png` - fitted parameters correlation ellipsesi

## Example

An example CSV file (`data.csv`) is provided with the script that produced it (`generate-fake-data.py`). The data represents measurements with uncertainties in both x and y variables.

Running the script with the provided example data will produce the following output:

```
Regression Results:
-----------------
Slope: 2.162649 ± 0.018603
Intercept: -0.521548 ± 0.147845

Covariance matrix:
 2.222766e-04 -1.474776e-03
-1.474776e-03  1.403852e-02
Pearson's Correlation coefficient: -0.536201

Chi-square: 20.241101
Degrees of freedom: 13
Reduced chi-square: 1.557008
P-value: 0.089359
```
These results will be saved as a text file. Three figures will also be saved, showing the data points with error bars and the best-fit line, residuals plot, and parameter correlation ellipses for the 1σ, 2σ, and 3σ confidence levels.

![fit plot](https://github.com/hezy/ODR-example/blob/main/fit_plot.png?raw=true)
![residuals plot](https://github.com/hezy/ODR-example/blob/main/residuals_plot.png?raw=true)
![correlation ellipses](https://github.com/hezy/ODR-example/blob/main/correlation_ellipses.png?raw=true)
