"""Generate synthetic data points with uncertainties following a quadratic trend.

This script creates a dataset of n points following a quadratic relationship
y = 2x + 0.01x² with added Gaussian noise. Both x and y coordinates have
associated uncertainties (dx, dy) drawn from normal distributions.

Data generation details:
- x: Evenly spaced values from 0 to n-1
- dx: Positive uncertainties, normal distribution (μ=0.1, σ=0.01)
- y: Follows 2x + 0.01x² with added Gaussian noise (σ=0.2)
- dy: Positive uncertainties, normal distribution (μ=0.15, σ=0.025)

Output:
- Saves data to 'data.csv' with columns: x, dx, y, dy
- All values are saved as floats with 4 decimal precision
- CSV includes header row and uses comma as delimiter
"""

import numpy as np

n = 15
x = np.arange(n, dtype=float)
dx = np.random.normal(loc=0.1, scale=0.01, size=n)
dx = np.clip(dx, 0, None)  # Ensure all values are positive
y = 2.0 * x + 0.01 * x**2 + 0.2 * (np.random.randn(n))
dy = np.random.normal(loc=0.15, scale=0.025, size=n)
dy = np.clip(dy, 0, None)  # Ensure all values are positive

# Save to CSV
np.savetxt(
    "data.csv",
    np.column_stack([x, dx, y, dy]),
    delimiter=",",
    header="x,dx,y,dy",
    comments="",
    fmt="%.4f",  # Use same float format for all columns
)
