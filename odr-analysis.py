import numpy as np
import pandas as pd
from scipy import odr
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def linear_func(p, x):
    """Linear function for ODR fitting: y = mx + b"""
    m, b = p
    return m * x + b


def read_data(filename):
    """Read data from CSV file"""
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


def confidence_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
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


def perform_odr(x, dx, y, dy):
    """Perform ODR analysis"""
    linear = odr.Model(linear_func)
    data = odr.RealData(x, y, sx=dx, sy=dy)
    odr_obj = odr.ODR(data, linear, beta0=[1.0, 0.0])
    results = odr_obj.run()

    n = len(x)
    p = 2  # number of parameters
    chi_square = results.sum_square
    chi_square_reduced = chi_square / (n - p)

    p_value = 1 - stats.chi2.cdf(chi_square, n - p)

    return results, chi_square, chi_square_reduced, p_value


def plot_results(x, dx, y, dy, results):
    """Create plots for data, fit, residuals, and correlation ellipse"""
    fig = plt.figure(figsize=(15, 15))
    gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 2])

    # Data and fit plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.errorbar(x, y, xerr=dx, yerr=dy, fmt="o", label="Data")

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = linear_func(results.beta, x_fit)

    ax1.plot(x_fit, y_fit, "r-", label="Fit")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("ODR Fit with Uncertainties")
    ax1.legend()
    ax1.grid(True)

    # Residuals plot
    ax2 = fig.add_subplot(gs[1, :])
    y_model = linear_func(results.beta, x)
    residuals = y - y_model

    total_uncertainty = np.sqrt(dy**2 + (results.beta[0] * dx) ** 2)

    ax2.errorbar(x, residuals, yerr=total_uncertainty, fmt="o")
    ax2.axhline(y=0, color="r", linestyle="-")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals")
    ax2.grid(True)

    # Correlation ellipse plot
    ax3 = fig.add_subplot(gs[2, :])

    confidence_data = [
        (2.30, 39.4, "red"),  # 1σ
        (6.18, 86.5, "green"),  # 2σ
        (11.83, 98.9, "blue"),  # 3σ
    ]

    for chi2_val, conf_level, color in confidence_data:
        confidence_ellipse(
            results.beta,
            results.cov_beta,
            ax3,
            n_std=np.sqrt(chi2_val),
            alpha=0.25,
            color=color,
            label=f"{conf_level:.1f}% confidence region",
        )

    ax3.plot(
        results.beta[0], results.beta[1], "r*", label="Best fit", markersize=10
    )

    ax3.set_xlabel("Slope (m)")
    ax3.set_ylabel("Intercept (b)")
    ax3.set_title("Parameter Correlation Ellipses (2D confidence regions)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    return fig


def main(filename):
    """Main function to run the analysis"""
    data = read_data(filename)
    if data is None:
        return

    x, dx, y, dy = data

    results, chi_square, chi_square_reduced, p_value = perform_odr(x, dx, y, dy)

    print("\nRegression Results:")
    print("-----------------")
    print(f"Slope: {results.beta[0]:.6f} ± {results.sd_beta[0]:.6f}")
    print(f"Intercept: {results.beta[1]:.6f} ± {results.sd_beta[1]:.6f}")
    print(
        f"\nCorrelation coefficient: {results.cov_beta[0,1] / (results.sd_beta[0] * results.sd_beta[1]):.6f}"
    )
    print(f"Chi-square: {chi_square:.6f}")
    print(f"Reduced chi-square: {chi_square_reduced:.6f}")
    print(f"P-value: {p_value:.6f}")

    fig = plot_results(x, dx, y, dy, results)
    plt.show()


if __name__ == "__main__":
    filename = "data.csv"
    main(filename)
