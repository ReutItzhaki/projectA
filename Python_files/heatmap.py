import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_dynamic_heatmap(matrix, vmin=None, vmax=None, cmap="viridis"):
    # Initialize the figure and axis for the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed
    
    # Divide the figure into grids to accommodate the plot, slider, and color scale
    grid = plt.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[19, 1])
    
    # Create the heatmap plot within the first grid
    ax = plt.subplot(grid[0, 0])
    
    # Flip the matrix along the vertical axis and use the specified colormap
    im = ax.imshow(np.flipud(matrix[0]), cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

    # Function to manually update the heatmap with keyboard/mouse input
    def update(event):
        nonlocal frame

        if event.key == 'right' and frame < len(matrix) - 1:
            frame += 1
        elif event.key == 'left' and frame > 0:
            frame -= 1

        im.set_data(np.flipud(matrix[frame]))
        ax.draw_artist(im)
        fig.canvas.blit(ax.bbox)

    # Function to handle mouse scroll events
    def on_scroll(event):
        nonlocal frame

        if event.button == 'up' and frame < len(matrix) - 1:
            frame += 1
        elif event.button == 'down' and frame > 0:
            frame -= 1

        im.set_data(np.flipud(matrix[frame]))
        ax.draw_artist(im)
        fig.canvas.blit(ax.bbox)
        fig.canvas.blit(ax.bbox)

    frame = 0
    fig.canvas.mpl_connect('key_press_event', update)
    fig.canvas.mpl_connect('scroll_event', on_scroll)  # Handle mouse scroll events

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Create a slider for frame selection and place it at the bottom of the figure
    slider_ax = plt.subplot(grid[1, 0])
    slider = Slider(slider_ax, 'Cycle', 0, len(matrix) - 1, valinit=0, valstep=1)

    def update_slider(val):
        nonlocal frame
        frame = int(slider.val)
        im.set_data(np.flipud(matrix[frame]))
        ax.draw_artist(im)
        fig.canvas.blit(ax.bbox)

    slider.on_changed(update_slider)

    # Add a color scale (colorbar) to the right of the heatmap plot
    cax = plt.subplot(grid[0, 1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Color Scale and Values')

    plt.show()

# # Example usage:
# # Create a 3D matrix of any dimensions
# matrix = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                   [[9, 8, 7], [6, 5, 4], [3, 2, 1]]])

# # Define vmin and vmax based on the min and max values in the entire matrix
# vmin = np.min(matrix)
# vmax = np.max(matrix)

# # Specify the desired colormap
# cmap = "viridis"  # Change this to any colormap you prefer

# plot_dynamic_heatmap(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
