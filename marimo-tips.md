# Marimo Notebook Tips & Best Practices

This document contains key lessons learned while working with marimo notebooks, particularly around plotting and reactive cell behavior.

## Reactive Cell System

### How Marimo Reactivity Works
- Marimo statically analyzes each cell to determine variable references and definitions
- Creates a directed acyclic graph (DAG) of cell dependencies
- When a cell runs, marimo automatically runs all other cells that reference any of the global variables it defines
- Cells run based on variable relationships, not page order
- Global variable names must be unique across cells

### Key Rules
- Mutations to variables do not trigger re-runs
- Deleting a cell removes its global variables
- Code and outputs remain consistent through automatic updates

## Plotting in Marimo

### Basic Plot Display
To display a matplotlib plot, include the `Axes` or `Figure` object as the last expression in the cell:

```python
@app.cell
def plot_example(plt):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax  # This displays the plot
    return
```

### Alternative Display Methods
- Use `plt.show()` or `fig.show()` to output in console area
- For interactive plots: `mo.mpl.interactive()` (note: matplotlib plots are not yet reactive)

### Common Plotting Mistakes

#### ❌ Wrong: Using conditionals that prevent execution
```python
@app.cell
def plot_data(x, y):
    if x is not None and y is not None:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax
    return  # Plot won't show if condition fails
```

#### ✅ Correct: Execute unconditionally with safe fallbacks
```python
@app.cell
def plot_data(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('X' if x is not None else 'Data')
    ax  # Always displays, handles None gracefully
    return
```

#### ❌ Wrong: Using return statements in cell body
```python
@app.cell
def plot_data(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    return ax  # ERROR: 'return' outside function
```

#### ✅ Correct: Last expression displays automatically
```python
@app.cell
def plot_data(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax  # This displays the plot
    return
```

## Cell Function Naming

### Named Functions vs Anonymous Functions

#### Named functions (recommended for most cases)
```python
@app.cell
def plot_results(data, plt):
    # Function executes when dependencies change
    fig, ax = plt.subplots()
    ax.plot(data)
    ax
    return
```

#### Anonymous functions (use with caution)
```python
@app.cell
def _(data, plt):
    # Also works, but less clear for debugging
    fig, ax = plt.subplots()
    ax.plot(data)
    ax
    return
```

## Variable Dependencies

### Function Parameters Define Dependencies
The parameters in your cell function determine which variables the cell depends on:

```python
@app.cell
def plot_analysis(x, y, results, column_names):
    # This cell will re-run when ANY of these variables change:
    # x, y, results, or column_names
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(column_names[0] if column_names else 'X')
    ax
    return
```

## Data Loading Patterns

### Flexible Column Handling
When working with CSV files that may have different column names:

```python
@app.cell
def read_flexible_csv(filename, pd):
    df = pd.read_csv(filename)

    # Use positional access instead of hardcoded names
    col_names = df.columns.tolist()
    x = df[col_names[0]].to_numpy()
    dx = df[col_names[1]].to_numpy()
    y = df[col_names[2]].to_numpy()
    dy = df[col_names[3]].to_numpy()

    return x, dx, y, dy, col_names
```

### Dynamic Labels with Fallbacks
```python
@app.cell
def plot_with_dynamic_labels(x, y, col_names, plt):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    # Use actual column names if available, fallback to generic
    if col_names is not None:
        ax.set_xlabel(col_names[0])
        ax.set_ylabel(col_names[2])
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    ax
    return
```

## Debugging Tips

### Check Cell Dependencies
- If a plot isn't updating, check that the plotting cell's function parameters include all necessary variables
- Use descriptive function names instead of `_` for easier debugging
- Remember that marimo runs cells based on variable dependencies, not cell order

### Variable Scope Issues
- Ensure global variable names are unique across all cells
- Use the `return` statement at the end of cells to explicitly define what variables are exported

### Plot Display Issues
- Make sure the plot object (ax, fig) is the last expression in the cell
- Avoid wrapping plot code in conditionals that might prevent execution
- Use safe fallbacks instead of preventing execution

## Performance Considerations

- Marimo's reactive system means expensive computations re-run when dependencies change
- Consider breaking expensive operations into separate cells to limit re-computation
- Use caching strategies for heavy data processing when appropriate

---

*These tips were compiled from practical experience debugging marimo plotting issues and understanding the reactive cell system.*