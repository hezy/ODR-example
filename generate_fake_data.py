import numpy as np

x = np.arange(15)
dx = np.full_like(x, 0.1)
y = (
    2 * x + 0.02 * x**2 + 0.2 * (np.random.randn(15))
)  # More accurate normal distribution
dy = 0.15 + 0.1 * np.random.rand(15)

# Save to CSV
np.savetxt(
    "data.csv",
    np.column_stack([x, dx, y, dy]),
    delimiter=",",
    header="x,dx,y,dy",
    comments="",
    fmt=["%d", "%.1f", "%.4f", "%.4f"],
)
