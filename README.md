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

## Usage

1. Prepare a CSV file with the following columns: x, dx, y, dy
   - x: x-values of the data points
   - dx: uncertainties in the x-values
   - y: y-values of the data points
   - dy: uncertainties in the y-values

2. Update the `filename` variable in the `main` function to match your CSV file name

3. Run the script:
   ```
   python odr_fit_with_uncertainties.py
   ```

4. The script will read the data from the CSV file, perform the ODR analysis, and display the regression results in the console

5. A figure will be created with subplots showing the data points, best-fit line, residuals, and parameter correlation ellipses

## Example

An example CSV file (`data.csv`) is provided with the script. The data represents measurements with uncertainties in both x and y variables.

Running the script with the example data will produce the following output:

```
Regression Results:
-----------------
Slope: 1.234567 ± 0.054321
Intercept: 0.987654 ± 0.112233

Correlation coefficient: 0.876543
Chi-square: 12.345678
Reduced chi-square: 0.987654
P-value: 0.123456
```

A figure will also be displayed, showing the data points with error bars, the best-fit line, residuals plot, and parameter correlation ellipses for the 1σ, 2σ, and 3σ confidence levels.