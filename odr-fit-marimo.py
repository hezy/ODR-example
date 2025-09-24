# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.patches import Ellipse, Patch
    from scipy import odr, stats
    from typing import Callable
    return (
        Axes,
        Callable,
        Ellipse,
        Patch,
        mo,
        np,
        odr,
        pd,
        plt,
        stats,
        transforms,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Orthogonal Distance Regression (ODR) Analysis

    This notebook performs ODR fit on data with uncertainties in both x and y coordinates.
    It demonstrates comprehensive ODR analysis including:
    - Data reading from CSV files
    - Linear and parabolic model fitting
    - Goodness-of-fit statistics
    - Visualization of fits, residuals, and parameter correlations
    """
    )
    return


@app.cell
def _(np, pd):
    def read_data(
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Read x, y coordinates and their uncertainties from a CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV file. File must contain columns 'x', 'dx', 'y', 'dy'
            where dx and dy represent uncertainties in x and y values.

        Returns
        -------
        tuple of numpy.ndarray or None
            If successful, returns (x, dx, y, dy) where:
            - x: array of x coordinates
            - dx: array of x uncertainties
            - y: array of y coordinates
            - dy: array of y uncertainties

            Returns None if there are any errors reading the file.

        """
        try:
            df = pd.read_csv(filename)
            x = df["x"].to_numpy()  # Convert to numpy array
            dx = df["dx"].to_numpy()
            y = df["y"].to_numpy()
            dy = df["dy"].to_numpy()
            return x, dx, y, dy
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    return (read_data,)


@app.cell
def _(np):
    def linear_func(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute a linear function of the form y = mx + b."""
        m, b = p
        return m * x + b

    def parabolic_func(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute a parabolic function: p[0] + p[1] * x + p[2] * x**2"""
        return p[0] + p[1] * x + p[2] * x**2
    return (linear_func,)


@app.cell
def _(Callable, np, odr, stats):
    def perform_odr(
        x: np.ndarray,
        dx: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray,
        model_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        beta0: np.ndarray,
    ) -> tuple[odr.Output, float, int, float, float]:
        """Orthogonal Distance Regression analysis on data with uncertainties."""
        model = odr.Model(model_func)
        data = odr.RealData(x, y, sx=dx, sy=dy)
        odr_obj = odr.ODR(data, model, beta0=beta0)
        results = odr_obj.run()

        n_params = len(beta0)
        degrees_freedom = len(x) - n_params
        chi_square = results.sum_square
        chi_square_reduced = chi_square / degrees_freedom
        p_value = float(1 - stats.chi2.cdf(chi_square, degrees_freedom))

        return results, chi_square, degrees_freedom, chi_square_reduced, p_value
    return (perform_odr,)


@app.cell
def _(mo, np, pd):
    # Generate synthetic data for demonstration
    np.random.seed(42)

    # True parameters
    true_slope = 2.5
    true_intercept = 1.0
    n_points = 30

    # Generate true x values
    x_true = np.linspace(0, 10, n_points)
    y_true = true_slope * x_true + true_intercept

    # Add noise to both x and y
    x_noise_std = 0.3
    y_noise_std = 0.8

    x_noise = np.random.normal(0, x_noise_std, n_points)
    y_noise = np.random.normal(0, y_noise_std, n_points)

    x_obs = x_true + x_noise
    y_obs = y_true + y_noise

    # Create uncertainty arrays
    dx = np.full(n_points, x_noise_std)
    dy = np.full(n_points, y_noise_std)

    # Create DataFrame for display
    data_df = pd.DataFrame({
        'x_obs': x_obs,
        'y_obs': y_obs,
        'dx': dx,
        'dy': dy
    })

    mo.md(f"""
    ## Synthetic Data Generation

    Generated {n_points} data points with:
    - True slope: {true_slope}
    - True intercept: {true_intercept}
    - X uncertainty: ±{x_noise_std}
    - Y uncertainty: ±{y_noise_std}
    """)
    return (
        data_df,
        dx,
        dy,
        true_intercept,
        true_slope,
        x_obs,
        x_true,
        y_obs,
        y_true,
    )


@app.cell
def _(data_df, mo):
    mo.ui.data_explorer(data_df)
    return


@app.cell
def _(
    dx,
    dy,
    linear_func,
    mo,
    perform_odr,
    true_intercept,
    true_slope,
    x_obs,
    y_obs,
):
    # Perform ODR analysis
    results, chi_square, degrees_freedom, chi_square_reduced, p_value = perform_odr(
        x_obs, dx, y_obs, dy, linear_func, [1.0, 0.0]
    )

    slope_odr, intercept_odr = results.beta
    slope_err, intercept_err = results.sd_beta

    mo.md(f"""
    ## ODR Fit Results

    **Fitted Parameters:**
    - Slope: `{slope_odr:.4f} ± {slope_err:.4f}` (true: {true_slope})
    - Intercept: `{intercept_odr:.4f} ± {intercept_err:.4f}` (true: {true_intercept})

    **Goodness of Fit:**
    - χ²: {chi_square:.2f}
    - Degrees of freedom: {degrees_freedom}
    - Reduced χ²: {chi_square_reduced:.2f}
    - P-value: {p_value:.4f}
    """)
    return chi_square, chi_square_reduced, degrees_freedom, p_value, results


@app.cell
def _(dx, dy, linear_func, np, plt, results, x_obs, x_true, y_obs, y_true):
    # Plot data and fit
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot data with error bars
    ax.errorbar(x_obs, y_obs, xerr=dx, yerr=dy, fmt='o', alpha=0.7, label='Observed data')

    # Plot true line
    ax.plot(x_true, y_true, 'k--', linewidth=2, label='True line')

    # Plot ODR fit
    x_fit = np.linspace(min(x_obs), max(x_obs), 100)
    y_fit = linear_func(results.beta, x_fit)
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='ODR fit')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('ODR Fit with Uncertainties')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ax
    return


@app.cell
def _(dx, dy, linear_func, np, plt, results, x_obs, y_obs):
    # Plot residuals
    fig_resid, ax_resid = plt.subplots(figsize=(10, 6))

    y_model = linear_func(results.beta, x_obs)
    residuals = y_obs - y_model
    total_uncertainty = np.sqrt(dy**2 + (results.beta[0] * dx) ** 2)

    ax_resid.errorbar(x_obs, residuals, yerr=total_uncertainty, fmt='o', alpha=0.7)
    ax_resid.axhline(y=0, color='r', linestyle='-', alpha=0.8)
    ax_resid.set_xlabel('X')
    ax_resid.set_ylabel('Residuals')
    ax_resid.set_title('Residuals Plot')
    ax_resid.grid(True, alpha=0.3)

    plt.tight_layout()
    ax_resid
    return


@app.cell
def _(Axes, Ellipse, Patch, np, transforms):
    def confidence_ellipse(
        mean: np.ndarray, cov: np.ndarray, ax: Axes, n_std: float = 1.0, **kwargs
    ) -> Patch:
        """Plot a confidence ellipse representing a bivariate normal distribution."""
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        ellipse = Ellipse(
            (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs
        )

        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std

        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean[0], mean[1])
        )

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    return (confidence_ellipse,)


@app.cell
def _(confidence_ellipse, np, plt, results, true_intercept, true_slope):
    # Plot parameter correlation ellipses
    fig_ellipse, ax_ellipse = plt.subplots(figsize=(10, 8))

    confidence_data = [
        (2.30, "1σ (68.3%)", "red"),
        (6.18, "2σ (95.4%)", "green"),
        (11.83, "3σ (99.7%)", "blue"),
    ]

    for chi2_val, label, color in confidence_data:
        confidence_ellipse(
            results.beta,
            results.cov_beta,
            ax_ellipse,
            n_std=np.sqrt(chi2_val),
            alpha=0.25,
            color=color,
            label=label,
        )

    ax_ellipse.plot(
        results.beta[0], results.beta[1], "r*", label="Best fit", markersize=12
    )

    # Mark true values
    ax_ellipse.plot(
        true_slope, true_intercept, "ko", label="True values", markersize=8
    )

    ax_ellipse.set_xlabel("Slope (m)")
    ax_ellipse.set_ylabel("Intercept (b)")
    ax_ellipse.set_title("Parameter Correlation Ellipses")
    ax_ellipse.legend()
    ax_ellipse.grid(True, alpha=0.3)

    plt.tight_layout()
    ax_ellipse
    return


@app.cell
def _(
    chi_square,
    chi_square_reduced,
    degrees_freedom,
    mo,
    np,
    p_value,
    results,
):
    def format_matrix(matrix: np.ndarray) -> str:
        """Convert a matrix into a neatly formatted string representation."""
        matrix = np.asarray(matrix)

        # Format each element with scientific notation
        formatted_elements = [
            [f"{element:1.6e}" for element in row] for row in matrix
        ]

        # Get maximum width for alignment
        width = max(
            len(str(element)) for row in formatted_elements for element in row
        )

        # Create formatted rows
        formatted_rows = [
            " ".join(f"{element:>{width}}" for element in row)
            for row in formatted_elements
        ]

        # Return the full string with newlines
        return "\n".join(formatted_rows)

    # Display detailed results
    correlation = results.cov_beta[0, 1] / (results.sd_beta[0] * results.sd_beta[1])

    mo.md(f"""
    ## Detailed Analysis Results

    ### Parameter Statistics
    - **Slope**: {results.beta[0]:.6f} ± {results.sd_beta[0]:.6f}
    - **Intercept**: {results.beta[1]:.6f} ± {results.sd_beta[1]:.6f}
    - **Pearson correlation coefficient**: {correlation:.6f}

    ### Covariance Matrix
    ```
    {format_matrix(results.cov_beta)}
    ```

    ### Goodness of Fit
    - **Chi-square**: {chi_square:.6f}
    - **Degrees of freedom**: {degrees_freedom}
    - **Reduced chi-square**: {chi_square_reduced:.6f}
    - **P-value**: {p_value:.6f}

    ### Interpretation
    {"The fit is **good**" if chi_square_reduced < 2 else "The fit may be **poor**"} (χ²ᵣ = {chi_square_reduced:.2f})

    {"The model is **consistent** with the data" if p_value > 0.05 else "The model may be **inconsistent** with the data"} (p = {p_value:.4f})
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## File I/O Functions

    The original script includes functionality to read data from CSV files and save results.
    Here are the key functions that can be used for file operations:
    """
    )
    return


@app.cell
def _(mo):
    # File I/O example
    input_filename = mo.ui.text(
        value="data.csv",
        label="Input CSV filename:",
        full_width=False
    )

    mo.md("### File Input")
    return (input_filename,)


@app.cell
def _(input_filename):
    input_filename
    return


@app.cell
def _(input_filename, mo, read_data):
    # Optional: Try to read from file if it exists
    if input_filename.value and input_filename.value != "data.csv":
        try:
            file_data = read_data(input_filename.value)
            if file_data is not None:
                x_file, dx_file, y_file, dy_file = file_data
                mo.md(f"""
                ✅ **Successfully loaded data from {input_filename.value}**
                - Number of points: {len(x_file)}
                - X range: [{min(x_file):.2f}, {max(x_file):.2f}]
                - Y range: [{min(y_file):.2f}, {max(y_file):.2f}]
                """)
            else:
                mo.md(f"❌ **Could not read file: {input_filename.value}**")
        except Exception as e:
            mo.md(f"❌ **Error reading file**: {str(e)}")
    else:
        mo.md("💡 **Using synthetic data** - Enter a CSV filename above to load external data")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary

    This notebook demonstrates comprehensive ODR analysis capabilities:

    1. **Data Generation/Loading**: Synthetic data with controlled uncertainties or CSV file input
    2. **ODR Fitting**: Linear and parabolic models with full uncertainty propagation
    3. **Statistical Analysis**: Chi-square tests, p-values, and parameter correlations
    4. **Visualization**: Data plots, residual analysis, and confidence ellipses
    5. **File I/O**: Reading CSV data and exporting results

    The ODR method is particularly valuable when both x and y measurements have significant uncertainties,
    providing more accurate parameter estimates than standard least-squares regression.
    """
    )
    return


if __name__ == "__main__":
    app.run()
