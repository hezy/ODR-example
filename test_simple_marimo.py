import marimo

__generated_with = "0.9.34"
app = marimo.App()

@app.cell
def __(mo):
    mo.md("# Simple ODR Test")
    return

@app.cell
def __(mo):
    # Simple file input
    filename_input = mo.ui.text(
        value="data.csv",
        label="CSV filename:"
    )
    return filename_input,

@app.cell
def __(filename_input, mo):
    # Display the input and try to load data
    filename = filename_input.value

    import pandas as pd
    import numpy as np

    try:
        # Try to load the CSV file
        df = pd.read_csv(filename)

        if all(col in df.columns for col in ['x', 'dx', 'y', 'dy']):
            x, dx, y, dy = df['x'].values, df['dx'].values, df['y'].values, df['dy'].values
            status_text = f"✅ Successfully loaded {len(x)} data points from {filename}"
        else:
            status_text = f"❌ Missing required columns in {filename}"
            x = dx = y = dy = None
    except Exception as e:
        status_text = f"❌ Error loading {filename}: {e}"
        x = dx = y = dy = None

    return mo.vstack([
        filename_input,
        mo.md(status_text)
    ]), x, dx, y, dy

@app.cell
def __(x, y, mo, plt, np):
    if x is not None and y is not None:
        # Create a simple plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Data Plot')

        return fig
    else:
        return mo.md("*Plot will appear when data is loaded*")

@app.cell
def __():
    import matplotlib.pyplot as plt
    import numpy as np
    return np, plt

@app.cell
def __():
    import marimo as mo
    return mo,

if __name__ == "__main__":
    app.run()