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

An example CSV file (`data.csv`) is provided with the script that produced it (`generate-fake-data.py`). The data represents measurements with uncertainties in both x and y variables.

Running the script with the provided example data will produce the following output:

```
Regression Results:
-----------------
Slope: 2.277464 ± 0.022014
Intercept: -0.592683 ± 0.182644

Correlation coefficient: -0.542123
Chi-square: 20.284218
Reduced chi-square: 1.560324
P-value: 0.088346
   
```
These results will be saved as a text file. Three figures will also be saved, showing the data points with error bars and the best-fit line, residuals plot, and parameter correlation ellipses for the 1σ, 2σ, and 3σ confidence levels.

![fit-plot](https://github.com/[hezy]/[ODR-example]/blob/[main]/fit-plot.png?raw=true)
![residuals-plot](https://github.com/[hezy]/[ODR-example]/blob/[main]/residuals-plot.png?raw=true)
![correlation-ellipses](https://github.com/[hezy]/[ODR-example]/blob/[main]/correlation-ellipses.png?raw=true)
