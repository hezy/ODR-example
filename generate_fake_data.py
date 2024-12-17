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
    fmt=["%d", "%.4f", "%.4f", "%.4f"],
)
