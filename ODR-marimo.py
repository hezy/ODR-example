# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "ai"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.odr import ODR, Model, RealData
    
    return

@app.cell
def _():
    mo.md("""
    # Linear Regression using Orthogonal Distance Regression (ODR)
    
    This notebook demonstrates linear regression using ODR, which accounts for errors in both the independent (x) and dependent (y) variables.
    """)
    return

@app.cell
def _():
    # Generate synthetic data with noise in both x and y
    np.random.seed(42)
    true_slope = 2.5
    true_intercept = 1.0
    n_points = 50
    
    x_true = np.linspace(0, 10, n_points)
    y_true = true_slope * x_true + true_intercept
    
    # Add noise to both x and y
    x_noise = np.random.normal(0, 0.5, n_points)
    y_noise = np.random.normal(0, 1.0, n_points)
    
    x_obs = x_true + x_noise
    y_obs = y_true + y_noise
    
    data_df = pd.DataFrame({'x_obs': x_obs, 'y_obs': y_obs})
    data_df.head()
    return

@app.cell
def _():
    mo.ui.data_explorer(data_df)
    return

@app.cell
def _():
    # Define the linear model for ODR
    def linear_func(B, x):
        return B[0] * x + B[1]
    
    # Prepare data for ODR
    model = Model(linear_func)
    data = RealData(x_obs, y_obs, sx=0.5, sy=1.0)
    odr = ODR(data, model, beta0=[1.0, 0.0])
    output = odr.run()
    
    # Extract fit parameters
    slope_odr, intercept_odr = output.beta
    slope_odr, intercept_odr
    return

@app.cell
def _():
    mo.md(f"""
    **ODR Fit Results:**
    
    - Slope: `{slope_odr:.3f}`
    - Intercept: `{intercept_odr:.3f}`
    """)
    return

@app.cell
def _():
    # Plot data and ODR fit
    plt.figure(figsize=(8, 6))
    plt.errorbar(x_obs, y_obs, xerr=0.5, yerr=1.0, fmt='o', label='Observed data', alpha=0.7)
    plt.plot(x_true, y_true, 'k--', label='True line')
    plt.plot(x_true, slope_odr * x_true + intercept_odr, 'r-', label='ODR fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression using ODR')
    plt.legend()
    plt.gca()
    return

if __name__ == "__main__":
    app.run()
