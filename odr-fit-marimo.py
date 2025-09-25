"""odr-fit-marimo.py performs ODR fit on data with uncertainties in both x and y.

This is the marimo notebook version of odr-fit.py. It provides an interactive
environment for exploring Orthogonal Distance Regression (ODR) analysis with
real-time visualization and educational content.
"""

# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def imports():
    """Import all required modules and libraries."""
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
        sys,
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
def data_io_functions(np, pd):
    """Define data input/output functions."""
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
def model_functions(np):
    """Define model functions for fitting."""
    def linear_func(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute a linear function of the form y = mx + b.

        Parameters
        ----------
        p : array-like, shape (2,)
            Parameters of the linear function:
            - p[0] (m): slope
            - p[1] (b): y-intercept
        x : array-like
            Independent variable values

        Returns
        -------
        array-like
            Computed y values: m*x + b

        """
        m, b = p
        return m * x + b

    def parabolic_func(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute a parabolic function.

        Parameters
        ----------
        p : array-like, shape (3,)
            Parameters of the parabolic function:
        x : array-like
            Independent variable values

        Returns
        -------
        array-like
            Computed y values: p[0] + p[1] * x + p[2] * x**2

        """
        return p[0] + p[1] * x + p[2] * x**2
    return (linear_func, parabolic_func)


@app.cell
def odr_analysis_functions(Callable, np, odr, stats):
    """Define ODR analysis functions."""
    def perform_odr(
        x: np.ndarray,
        dx: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray,
        model_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        beta0: np.ndarray,
    ) -> tuple[odr.Output, float, int, float, float]:
        """Orthogonal Distance Regression analysis on data with uncertainties.

        Fits a given model function to data points with uncertainties in both x and y
        using ODR method from scipy.odr. Also computes goodness-of-fit statistics.

        Parameters
        ----------
        x : array-like
            X coordinates of the data points
        dx : array-like
            Uncertainties (standard deviations) in x coordinates
        y : array-like
            Y coordinates of the data points
        dy : array-like
            Uncertainties (standard deviations) in y coordinates
        model_func : callable
            Model function to fit. Should take parameters (beta, x) where beta is
            the parameter vector and x is the independent variable array
        beta0 : array-like
            Initial guesses for the model parameters

        Returns
        -------
        results : ODR
            ODR result object containing fit parameters and covariance matrix
        chi_square : float
            Chi-square statistic of the fit
        degrees_freedom : int
            Number of degrees of freedom (n_points - n_parameters)
        chi_square_reduced : float
            Reduced chi-square (chi-square / degrees_freedom)
        p_value : float
            P-value for the chi-square goodness-of-fit test
        """
        model = odr.Model(model_func)
        data = odr.RealData(x, y, sx=dx, sy=dy)
        odr_obj = odr.ODR(data, model, beta0=beta0)
        results = odr_obj.run()

        n_params = len(beta0)
        degrees_freedom = len(x) - n_params
        chi_square = results.sum_square  # type: ignore # ODR Output attribute exists at runtime
        chi_square_reduced = chi_square / degrees_freedom
        p_value = float(1 - stats.chi2.cdf(chi_square, degrees_freedom))

        return results, chi_square, degrees_freedom, chi_square_reduced, p_value
    return (perform_odr,)




@app.cell
def load_default_data(mo, pd, read_data):
    """Load the default CSV file for analysis."""
    try:
        # Load the default data.csv file
        file_data = read_data("data.csv")
        if file_data is not None:
            x, dx, y, dy = file_data

            # Create DataFrame for display
            data_df = pd.DataFrame({
                'x': x,
                'y': y,
                'dx': dx,
                'dy': dy
            })

            mo.md(f"""
            ## Data Loaded Successfully

            Loaded {len(x)} data points from 'data.csv':
            - X range: [{min(x):.2f}, {max(x):.2f}]
            - Y range: [{min(y):.2f}, {max(y):.2f}]
            - Average X uncertainty: {dx.mean():.3f}
            - Average Y uncertainty: {dy.mean():.3f}

            This uses the same data loading as the CLI version.
            """)
        else:
            mo.md("❌ Could not load data.csv - make sure the file exists")
            data_df, dx, dy, x, y = None, None, None, None, None
    except Exception as e:
        mo.md(f"❌ Error loading data.csv: {e}")
        data_df, dx, dy, x, y = None, None, None, None, None

    return data_df, dx, dy, x, y


@app.cell
def display_data_explorer(data_df, mo):
    """Display interactive data explorer."""
    if data_df is not None:
        mo.ui.data_explorer(data_df)
    else:
        mo.md("*Data explorer will appear when data is loaded*")
    return


@app.cell
def perform_linear_odr(
    dx,
    dy,
    linear_func,
    mo,
    perform_odr,
    x,
    y,
):
    """Perform ODR analysis on the data."""
    if x is not None and y is not None:
        # Perform ODR analysis using the same approach as CLI version
        results, chi_square, degrees_freedom, chi_square_reduced, p_value = perform_odr(
            x, dx, y, dy, linear_func, [1.0, 0.0]
        )

        slope_odr, intercept_odr = results.beta
        slope_err, intercept_err = results.sd_beta

        mo.md(f"""
        ## ODR Analysis Results

        **Fitted Parameters:**
        - Slope: `{slope_odr:.4f} ± {slope_err:.4f}`
        - Intercept: `{intercept_odr:.4f} ± {intercept_err:.4f}`

        **Goodness of Fit:**
        - χ²: {chi_square:.2f}
        - Degrees of freedom: {degrees_freedom}
        - Reduced χ²: {chi_square_reduced:.2f}
        - P-value: {p_value:.4f}

        **Interpretation:**
        {'✅ Good fit' if chi_square_reduced < 2 else '⚠️ Poor fit'} (χ²ᵣ = {chi_square_reduced:.2f})
        {'✅ Model consistent with data' if p_value > 0.05 else '⚠️ Model may be inconsistent'} (p = {p_value:.4f})
        """)
    else:
        mo.md("*ODR analysis results will appear when data is loaded*")
        chi_square, chi_square_reduced, degrees_freedom, p_value, results = None, None, None, None, None

    return chi_square, chi_square_reduced, degrees_freedom, p_value, results


@app.cell
def plot_fit_results(dx, dy, linear_func, np, plt, results, x, y):
    """Create and display plot of data points with error bars and fit line."""
    # This mirrors the plot_fit function from the CLI version
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine if error bars are visible (same logic as CLI version)
    median_dx = np.median(dx)
    median_dy = np.median(dy)
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)

    _use_points_fit = (median_dx / x_range < 0.01) and (median_dy / y_range < 0.01)
    _marker_fit = "o" if _use_points_fit else "o"  # Always show points in marimo for clarity

    # Plot data with error bars
    ax.errorbar(x, y, xerr=dx, yerr=dy, fmt=_marker_fit, alpha=0.7, label='Data')

    # Plot ODR fit
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = linear_func(results.beta, x_fit)
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fit')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('ODR Fit with Uncertainties')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    ax
    return


@app.cell
def plot_residuals_analysis(dx, dy, linear_func, np, plt, results, x, y):
    """Generate and display residuals plot for the linear fit."""
    # This mirrors the plot_residuals function from the CLI version
    fig_resid, ax_resid = plt.subplots(figsize=(10, 6))

    y_model = linear_func(results.beta, x)
    residuals = y - y_model
    total_uncertainty = np.sqrt(dy**2 + (results.beta[0] * dx) ** 2)

    # Determine if error bars are visible (same logic as CLI version)
    median_uncert = np.median(total_uncertainty)
    resid_range = np.max(residuals) - np.min(residuals)
    _use_points_resid = median_uncert / resid_range < 0.01
    _marker_resid = "o" if _use_points_resid else "o"  # Always show points in marimo for clarity

    ax_resid.errorbar(x, residuals, yerr=total_uncertainty, fmt=_marker_resid, alpha=0.7)
    ax_resid.axhline(y=0, color='r', linestyle='-')
    ax_resid.set_xlabel('X')
    ax_resid.set_ylabel('Residuals')
    ax_resid.set_title('Residuals')
    ax_resid.grid(True)

    plt.tight_layout()
    ax_resid
    return


@app.cell
def plotting_utilities(Axes, Ellipse, Patch, np, transforms):
    """Define utility functions for plotting."""
    def confidence_ellipse(
        mean: np.ndarray, cov: np.ndarray, ax: Axes, n_std: float = 1.0, **kwargs
    ) -> Patch:
        """Plot a confidence ellipse representing a bivariate normal distribution.

        This function creates an ellipse that visualizes the covariance structure
        and mean of a 2D normally distributed dataset. The ellipse's size represents
        the confidence interval determined by n_std standard deviations.

        Parameters
        ----------
        mean : array-like, shape (2,)
            The center point (mean) of the ellipse in format [x, y]
        cov : array-like, shape (2, 2)
            The 2x2 covariance matrix of the distribution
        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse on
        n_std : float, optional (default=1.0)
            The number of standard deviations determining the ellipse's size
        **kwargs : dict
            Additional keyword arguments passed to matplotlib.patches.Ellipse

        Returns
        -------
        matplotlib.patches.Ellipse
            The added ellipse patch object

        """
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
def plot_correlation_ellipses(confidence_ellipse, np, plt, results):
    """Create and display plot showing parameter correlation ellipses."""
    # This mirrors the plot_ellipses function from the CLI version
    fig_ellipse, ax_ellipse = plt.subplots(figsize=(10, 8))

    confidence_data = [
        (2.30, "1σ (39.3%)", "red"),  # Corrected percentages to match CLI
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
        results.beta[0], results.beta[1], "r*", label="Best fit", markersize=10
    )

    ax_ellipse.set_xlabel("Slope (m)")
    ax_ellipse.set_ylabel("Intercept (b)")
    ax_ellipse.set_title("Parameter Correlation Ellipses")
    ax_ellipse.legend()
    ax_ellipse.grid(True)

    plt.tight_layout()
    ax_ellipse
    return


@app.cell
def utility_functions_and_results(
    chi_square,
    chi_square_reduced,
    degrees_freedom,
    mo,
    np,
    p_value,
    results,
):
    """Define utility functions and display detailed results."""
    def format_matrix(matrix: np.ndarray) -> str:
        """Convert a matrix into a neatly formatted string representation.

        Converts each element to scientific notation with 6 decimal places and
        aligns columns for readability.

        Parameters
        ----------
        matrix : array-like
            Input 2D matrix that can be converted to a NumPy array.

        Returns
        -------
        str
            String representation of the matrix where:
            - Each element is in scientific notation (1.234567e+00 format)
            - Elements are right-aligned within columns
            - Rows are separated by newlines
            - Columns are separated by single spaces

        """
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

    # Display detailed results (mirrors the CLI version output format)
    correlation = results.cov_beta[0, 1] / (results.sd_beta[0] * results.sd_beta[1])

    mo.md(f"""
    ## Detailed Analysis Results

    This section mirrors the output that would be saved to 'fit_results.txt' in the CLI version.

    ### Regression Results
    - **Slope**: {results.beta[0]:.6f} ± {results.sd_beta[0]:.6f}
    - **Intercept**: {results.beta[1]:.6f} ± {results.sd_beta[1]:.6f}

    ### Covariance Matrix
    ```
    {format_matrix(results.cov_beta)}
    ```

    **Pearson's Correlation coefficient**: {correlation:.6f}

    ### Goodness of Fit Statistics
    - **Chi-square**: {chi_square:.6f}
    - **Degrees of freedom**: {degrees_freedom}
    - **Reduced chi-square**: {chi_square_reduced:.6f}
    - **P-value**: {p_value:.6f}

    ### Interpretation
    {"The fit is **good**" if chi_square_reduced < 2 else "The fit may be **poor**"} (χ²ᵣ = {chi_square_reduced:.2f})

    {"The model is **consistent** with the data" if p_value > 0.05 else "The model may be **inconsistent** with the data"} (p = {p_value:.4f})
    """)
    return (format_matrix,)


@app.cell
def file_io_section(mo):
    """Introduction to file I/O capabilities."""
    mo.md(
        """
    ## File I/O Functions

    The original CLI script includes functionality to read data from CSV files and save results.
    This marimo version can also read CSV files with the same format expected by the CLI version:
    - Column 'x': x coordinates
    - Column 'dx': x uncertainties
    - Column 'y': y coordinates
    - Column 'dy': y uncertainties
    """
    )
    return


@app.cell
def file_input_ui(mo):
    """Create file input UI element."""
    # File I/O example - matches the default from CLI version
    input_filename = mo.ui.text(
        value="data.csv",
        label="Input CSV filename (same format as CLI version):",
        full_width=False
    )

    mo.md("### File Input")
    return (input_filename,)


@app.cell
def display_filename_input(input_filename):
    """Display the filename input widget."""
    input_filename
    return


@app.cell
def handle_file_input(input_filename, mo, read_data):
    """Handle file input and provide feedback."""
    # Optional: Try to read from file if it exists (matches CLI behavior)
    if input_filename.value and input_filename.value != "data.csv":
        try:
            _file_data = read_data(input_filename.value)
            if _file_data is not None:
                x_file, dx_file, y_file, dy_file = _file_data
                mo.md(f"""
                ✅ **Successfully loaded data from {input_filename.value}**
                - Number of points: {len(x_file)}
                - X range: [{min(x_file):.2f}, {max(x_file):.2f}]
                - Y range: [{min(y_file):.2f}, {max(y_file):.2f}]

                *This data would be processed the same way as in the CLI version.*
                """)
            else:
                mo.md(f"❌ **Could not read file: {input_filename.value}**")
        except Exception as e:
            mo.md(f"❌ **Error reading file**: {str(e)}")
    else:
        mo.md("💡 **Enter a CSV filename above to load data** (same format as CLI version)")
    return


@app.cell
def summary_section(mo):
    """Provide summary of capabilities."""
    mo.md(
        """
    ## Summary

    This marimo notebook demonstrates the same ODR analysis capabilities as the CLI version:

    1. **Data Loading**: CSV file input with same format as CLI version
    2. **ODR Analysis**: Same `perform_odr()` function with identical statistical calculations
    3. **Visualization**: Same plotting logic as CLI version but with inline display
    4. **Results**: Same detailed statistics and goodness-of-fit metrics
    5. **Educational Enhancement**: Interactive exploration of ODR analysis

    ### Key Similarities with CLI Version:
    - Identical core analysis functions (`read_data`, `perform_odr`, `confidence_ellipse`, etc.)
    - Same statistical calculations and goodness-of-fit tests
    - Same plotting logic (error bar visibility, axis labels, etc.)
    - Same result formatting and covariance matrix display

    ### Marimo Enhancements:
    - Interactive data exploration
    - Inline plot display
    - Educational markdown documentation

    The ODR method is particularly valuable when both x and y measurements have significant uncertainties,
    providing more accurate parameter estimates than standard least-squares regression.
    """
    )
    return


if __name__ == "__main__":
    app.run()
