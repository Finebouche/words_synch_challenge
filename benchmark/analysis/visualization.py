import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Function to get color from a Plotly colormap
def get_color_from_px_cmap(cmap_name, value):
    """Get the color from a Plotly Express colormap given a value between 0 and 1."""
    cmap = px.colors.get_colorscale(cmap_name)
    scaled_value = value * (len(cmap) - 1)  # Scale value to the colormap range
    index = int(np.floor(scaled_value))
    return cmap[index][1]  # Extract the color

# Function to create gradient lines using Plotly Express colormaps
def create_px_colormap_gradient_lines(embeddings, num_points, cmap_name):
    lines = []
    for i in range(num_points - 1):
        factor = i / (num_points - 1)  # Normalized position for gradient
        color = get_color_from_px_cmap(cmap_name, factor)
        line = go.Scatter3d(
            x=[embeddings[i, 0], embeddings[i + 1, 0]],
            y=[embeddings[i, 1], embeddings[i + 1, 1]],
            z=[embeddings[i, 2], embeddings[i + 1, 2]],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False
        )
        lines.append(line)
    return lines

# Function to create lines with fixed color (blue for player 1, red for player 2)
def create_fixed_color_lines(embeddings, num_points, color):
    """Create lines between embeddings with a fixed color."""
    lines = []
    for i in range(num_points - 1):
        line = go.Scatter3d(
            x=[embeddings[i, 0], embeddings[i + 1, 0]],
            y=[embeddings[i, 1], embeddings[i + 1, 1]],
            z=[embeddings[i, 2], embeddings[i + 1, 2]],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False
        )
        lines.append(line)
    return lines

