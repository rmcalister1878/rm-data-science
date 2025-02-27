import matplotlib.pyplot as plt

def grid_plots(cols):
    num_plots = len(cols)
    grid_rows = (num_plots // 3) + (num_plots % 3 > 0)  # Calculate number of rows needed

    # Create a grid of subplots
    fig, axes = plt.subplots(grid_rows, 3, figsize=(3 * 5, grid_rows * 5))
    axes = axes.flatten()
    return axes