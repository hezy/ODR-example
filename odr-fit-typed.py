"""odr-fit.py performs ODR fit on data with uncertainties in both the x and y.

It reads data from a CSV file, performs the Orthogonal Distance Regression
(ODR) analysis, and creates plots for the data with fit, residuals, and
parameter correlation ellipses.
"""

# pyright: ignore-all

import sys

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import odr, stats


def read_data(
    filename: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read data from CSV file."""
    try:
        df = pd.read_csv(filename)
        x = df["x"].values
        dx = df["dx"].values
        y = df["y"].values
        dy = df["dy"].values
        return x, dx, y, dy
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def linear_func(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear function for ODR fitting: y = mx + b"""
    m, b = p
    return m * x + b


def perform_odr(
    x: np.ndarray, dx: np.ndarray, y: np.ndarray, dy: np.ndarray
) -> tuple[odr.Output, float, int, float, float]:
    """Perform ODR analysis"""
    linear = odr.Model(linear_func)
    data = odr.RealData(x, y, sx=dx, sy=dy)
    odr_obj = odr.ODR(data, linear, beta0=[1.0, 0.0])
    results = odr_obj.run()

    degrees_freedom = len(x) - 2
    chi_square = results.sum_square
    chi_square_reduced = chi_square / degrees_freedom
    p_value = 1 - stats.chi2.cdf(chi_square, degrees_freedom)

    return results, chi_square, degrees_freedom, chi_square_reduced, p_value


def plot_fit(
    x: np.ndarray,
    dx: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    results: odr.Output,
    save_path: str,
) -> None:
    """Create and save fit plot"""
    fig = plt.figure(figsize=(10, 8))

    median_dx = np.median(dx)
    median_dy = np.median(dy)
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)

    use_points = (median_dx / x_range < 0.01) and (median_dy / y_range < 0.01)
    marker = "o" if use_points else "none"

    plt.errorbar(x, y, xerr=dx, yerr=dy, fmt=marker, label="Data")

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = linear_func(results.beta, x_fit)
    plt.plot(x_fit, y_fit, "r-", label="Fit")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("ODR Fit with Uncertainties")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


def plot_residuals(
    x: np.ndarray,
    dx: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    results: odr.Output,
    save_path: str,
) -> None:
    """Create and save residuals plot"""
    fig = plt.figure(figsize=(10, 6))

    y_model = linear_func(results.beta, x)
    residuals = y - y_model
    total_uncertainty = np.sqrt(dy**2 + (results.beta[0] * dx) ** 2)

    median_uncert = np.median(total_uncertainty)
    resid_range = np.max(residuals) - np.min(residuals)
    use_points = median_uncert / resid_range < 0.01
    marker = "o" if use_points else "none"

    plt.errorbar(x, residuals, yerr=total_uncertainty, fmt=marker)
    plt.axhline(y=0, color="r", linestyle="-")
    plt.xlabel("X")
    plt.ylabel("Residuals")
    plt.title("Residuals")
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


def confidence_ellipse(
    mean: np.ndarray,
    cov: np.ndarray,
    ax: plt.Axes,
    n_std: float = 1.0,
    **kwargs,
) -> Ellipse:
    """
    Create a plot of the covariance confidence ellipse of parameters m and b.
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


def format_matrix(matrix: np.ndarray) -> str:
    matrix = np.asarray(matrix)

    formatted_elements = [
        [f"{element:1.6e}" for element in row] for row in matrix
    ]

    width = max(
        len(str(element)) for row in formatted_elements for element in row
    )

    formatted_rows = [
        " ".join(f"{element:>{width}}" for element in row)
        for row in formatted_elements
    ]

    return "\n".join(formatted_rows)


def plot_ellipses(results: odr.Output, save_path: str) -> None:
    """Create and save correlation ellipse plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    confidence_data = [
        (2.30, "1σ", "red"),
        (6.18, "2σ", "green"),
        (11.83, "3σ", "blue"),
    ]

    for chi2_val, label, color in confidence_data:
        confidence_ellipse(
            results.beta,
            results.cov_beta,
            ax,
            n_std=np.sqrt(chi2_val),
            alpha=0.25,
            color=color,
            label=label,
        )

    ax.plot(
        results.beta[0], results.beta[1], "r*", label="Best fit", markersize=10
    )

    plt.xlabel("Slope (m)")
    plt.ylabel("Intercept (b)")
    plt.title("Parameter Correlation Ellipses")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


def main() -> None:
    """Main function to run the analysis"""
    default_input = "data.csv"

    if len(sys.argv) == 1:
        input_file = default_input
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
    else:
        print("Usage: odr-analysis [input_file.csv]")
        print(f"Default input file: {default_input}")
        sys.exit(1)

    data = read_data(input_file)
    if data is None:
        return

    x, dx, y, dy = data
    results, chi_square, degrees_freedom, chi_square_reduced, p_value = (
        perform_odr(x, dx, y, dy)
    )

    plot_fit(x, dx, y, dy, results, "fit_plot.png")
    plot_residuals(x, dx, y, dy, results, "residuals_plot.png")
    plot_ellipses(results, "correlation_ellipses.png")

    with open("fit_results.txt", "w") as f:
        f.write("Regression Results:\n")
        f.write("-----------------\n")
        f.write(f"Slope: {results.beta[0]:.6f} ± {results.sd_beta[0]:.6f}\n")
        f.write(
            f"Intercept: {results.beta[1]:.6f} ± {results.sd_beta[1]:.6f}\n"
        )
        f.write(f"\nCovariance matrix:")
        f.write(f"\n{format_matrix(results.cov_beta)}")
        f.write(
            f"\nPearson's Correlation coefficient: {results.cov_beta[0,1] / (results.sd_beta[0] * results.sd_beta[1]):.6f}\n"
        )
        f.write(f"\nChi-square: {chi_square:.6f}\n")
        f.write(f"Degrees of freedom: {degrees_freedom}\n")
        f.write(f"Reduced chi-square: {chi_square_reduced:.6f}\n")
        f.write(f"P-value: {p_value:.6f}\n")


if __name__ == "__main__":
    main()
