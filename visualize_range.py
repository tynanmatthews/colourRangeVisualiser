import plotly.graph_objects as go
import numpy as np
from range_from_swatch import extract_colour_range_from_swatch
import argparse

def rgb_to_hex(rgb):
    """Convert RGB values to hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def hsv_to_hex(hsv):
    """Convert HSV values to hex color string."""
    # Convert HSV to RGB (OpenCV format: H:0-180, S:0-255, V:0-255)
    h, s, v = hsv
    h = h * 2  # Convert to 0-360 range
    s = s / 255.0
    v = v / 255.0
    
    c = v * s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = v - c
    rgb = np.zeros(3)
    
    if 0 <= h < 60:
        rgb = [c, x, 0]
    elif 60 <= h < 120:
        rgb = [x, c, 0]
    elif 120 <= h < 180:
        rgb = [0, c, x]
    elif 180 <= h < 240:
        rgb = [0, x, c]
    elif 240 <= h < 300:
        rgb = [x, 0, c]
    else:
        rgb = [c, 0, x]
    
    rgb = [(r + m) * 255 for r in rgb]
    return rgb_to_hex(rgb)

def get_vertex_colors(vertices, color_space):
    """Get colors for each vertex."""
    if color_space == 'RGB':
        return [rgb_to_hex(v) for v in vertices]
    else:  # HSV
        return [hsv_to_hex(v) for v in vertices]

def visualize_color_range(color_range, color_space='RGB', to_scale=False):
    """
    Creates an interactive 3D visualization of a color range in the browser.
    
    Args:
        color_range: List of two points [[min_c1, min_c2, min_c3], [max_c1, max_c2, max_c3]]
                    representing the corners of the color range
        color_space: String indicating the color space ('RGB' or 'HSV')
        to_scale: Boolean indicating whether to show the full color space range
    """
    # Extract the min and max points
    min_point = np.array(color_range[0])
    max_point = np.array(color_range[1])
    
    # Create the 8 vertices of the color range cube
    vertices = np.array([
        [min_point[0], min_point[1], min_point[2]],  # 000
        [max_point[0], min_point[1], min_point[2]],  # 100
        [min_point[0], max_point[1], min_point[2]],  # 010
        [max_point[0], max_point[1], min_point[2]],  # 110
        [min_point[0], min_point[1], max_point[2]],  # 001
        [max_point[0], min_point[1], max_point[2]],  # 101
        [min_point[0], max_point[1], max_point[2]],  # 011
        [max_point[0], max_point[1], max_point[2]]   # 111
    ])

    # Get colors for each vertex
    vertex_colors = get_vertex_colors(vertices, color_space)

    # Define the 6 faces of the cube (each face is a list of 4 vertex indices)
    faces = [
        [0, 1, 3, 2],  # bottom face
        [4, 5, 7, 6],  # top face
        [0, 1, 5, 4],  # front face
        [2, 3, 7, 6],  # back face
        [0, 2, 6, 4],  # left face
        [1, 3, 7, 5]   # right face
    ]

    # Create the mesh for each face
    mesh_data = []
    for face in faces:
        # Get the vertices for this face
        face_vertices = vertices[face]
        face_colors = [vertex_colors[i] for i in face]
        
        # Create the mesh for this face
        mesh_data.append(
            go.Mesh3d(
                x=face_vertices[:, 0],
                y=face_vertices[:, 1],
                z=face_vertices[:, 2],
                i=[0],
                j=[1],
                k=[2],
                vertexcolor=face_colors,
                opacity=1,
                showscale=False,
                hoverinfo='skip'
            )
        )

    # Create the edges
    edges = np.array([
        [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ])

    # Create the lines for the cube edges
    x_lines = []
    y_lines = []
    z_lines = []
    
    for edge in edges:
        x_lines.extend([vertices[edge[0]][0], vertices[edge[1]][0], None])
        y_lines.extend([vertices[edge[0]][1], vertices[edge[1]][1], None])
        z_lines.extend([vertices[edge[0]][2], vertices[edge[1]][2], None])

    # Add the edges to the mesh data
    mesh_data.append(
        go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color='black', width=2),
            name='Color Range'
        )
    )

    # Create the 3D plot
    fig = go.Figure(data=mesh_data)

    # Set axis labels and ranges based on color space
    if color_space == 'RGB':
        axis_labels = ['Red', 'Green', 'Blue']
        if to_scale:
            axis_ranges = {
                'xaxis': {'range': [0, 255]},
                'yaxis': {'range': [0, 255]},
                'zaxis': {'range': [0, 255]}
            }
        else:
            # Add some padding to the range
            padding = 10
            axis_ranges = {
                'xaxis': {'range': [min_point[0] - padding, max_point[0] + padding]},
                'yaxis': {'range': [min_point[1] - padding, max_point[1] + padding]},
                'zaxis': {'range': [min_point[2] - padding, max_point[2] + padding]}
            }
    else:  # HSV
        axis_labels = ['Hue', 'Saturation', 'Value']
        if to_scale:
            axis_ranges = {
                'xaxis': {'range': [0, 180]},  # Hue in OpenCV is 0-180
                'yaxis': {'range': [0, 255]},  # Saturation
                'zaxis': {'range': [0, 255]}   # Value
            }
        else:
            # Add some padding to the range
            padding = 10
            axis_ranges = {
                'xaxis': {'range': [min_point[0] - padding, max_point[0] + padding]},
                'yaxis': {'range': [min_point[1] - padding, max_point[1] + padding]},
                'zaxis': {'range': [min_point[2] - padding, max_point[2] + padding]}
            }

    # Update layout
    fig.update_layout(
        title=f'Color Range Visualization in {color_space} Space',
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            aspectmode='cube',
            **axis_ranges
        ),
        showlegend=False
    )

    # Show the plot in the browser
    fig.show()

def process_and_visualize_swatch(file_path, config, to_scale=False):
    """
    Process a swatch image and visualize its color range.
    
    Args:
        file_path (str): Path to the swatch image
        config (dict): Configuration dictionary for color range extraction
        to_scale (bool): Whether to show the full color space range
    """
    try:
        # Extract color range
        color_range = extract_colour_range_from_swatch(file_path, config)
        
        # Visualize the range
        visualize_color_range(color_range, config['colour_space'], to_scale)
        
        return color_range
    except Exception as e:
        print(f"Error processing swatch: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a swatch image and visualize its color range')
    parser.add_argument('image_path', help='Path to the swatch image')
    parser.add_argument('--colour-space', choices=['RGB', 'HSV'], default='RGB',
                      help='Color space to use (default: RGB)')
    parser.add_argument('--discrim-bands', type=int, default=3,
                      help='Number of k-means clusters (default: 3)')
    parser.add_argument('--retention', type=float, default=50.0,
                      help='Percentage of points to retain (default: 50.0)')
    parser.add_argument('--to-scale', action='store_true',
                      help='Show the full color space range (0-255 for RGB, 0-180/255 for HSV)')
    
    args = parser.parse_args()
    
    config = {
        'colour_space': args.colour_space,
        'discrim_bands': args.discrim_bands,
        'retention': args.retention
    }
    
    try:
        color_range = process_and_visualize_swatch(args.image_path, config, args.to_scale)
        print(f"Successfully processed swatch. Color range: {color_range}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1) 