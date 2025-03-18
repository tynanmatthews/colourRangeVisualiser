import cv2
import numpy as np
import plotly.graph_objects as go
import argparse
from range_from_swatch import assert_input_colour_space
import os

def rgb_to_hex(rgb):
    """Convert RGB values to hex color string."""
    r, g, b = [max(0, min(255, int(val))) for val in rgb]  # Clamp values to valid range
    return f'rgb({r}, {g}, {b})'

def hsv_to_rgb(hsv):
    """Convert HSV values to RGB values."""
    # Ensure hsv values are within valid ranges
    h = max(0, min(180, float(hsv[0])))
    s = max(0, min(255, float(hsv[1])))
    v = max(0, min(255, float(hsv[2])))
    
    # Convert to uint8 for OpenCV
    hsv_pixel = np.array([[[h, s, v]]], dtype=np.uint8)
    rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)
    return rgb_pixel[0, 0].tolist()

def visualize_image_colors(image_path, color_space='RGB', sample_ratio=0.1, to_scale=False):
    """
    Visualize the colors of an image in 3D color space.
    
    Args:
        image_path (str): Path to the image
        color_space (str): Color space to use ('RGB' or 'HSV')
        sample_ratio (float): Ratio of pixels to sample (to avoid too many points for large images)
        to_scale (bool): Whether to show the full color space range
    """
    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    print(f"Image size: {img.shape}")
    
    # Convert to the specified color space
    img = assert_input_colour_space(img, color_space)
    
    # Reshape image to a list of pixels
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    print(f"Total pixels: {len(pixels)}")
    
    # Sample pixels to reduce number of points
    if sample_ratio < 1.0:
        np.random.seed(42)  # For reproducibility
        sample_size = max(1000, int(len(pixels) * sample_ratio))  # Ensure at least 1000 points
        indices = np.random.choice(len(pixels), size=sample_size, replace=False)
        pixels = pixels[indices]
    
    print(f"Sampled pixels: {len(pixels)}")
    
    # Create colors for markers
    if color_space == 'RGB':
        marker_colors = [rgb_to_hex(p) for p in pixels]
    else:  # HSV
        marker_colors = [rgb_to_hex(hsv_to_rgb(p)) for p in pixels]
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=pixels[:, 0],
            y=pixels[:, 1],
            z=pixels[:, 2],
            mode='markers',
            marker=dict(
                size=3,  # Increased marker size
                color=marker_colors,
                opacity=0.7
            ),
            hoverinfo='skip'
        )
    ])
    
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
            # Automatically set the range based on the data with safety limits
            padding = 20.0  # Increased padding
            min_vals = np.min(pixels, axis=0)
            max_vals = np.max(pixels, axis=0)
            
            # Print the range to help debug
            print(f"Data range - Min: {min_vals}, Max: {max_vals}")
            
            axis_ranges = {
                'xaxis': {'range': [max(0, min_vals[0] - padding), min(255, max_vals[0] + padding)]},
                'yaxis': {'range': [max(0, min_vals[1] - padding), min(255, max_vals[1] + padding)]},
                'zaxis': {'range': [max(0, min_vals[2] - padding), min(255, max_vals[2] + padding)]}
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
            # Automatically set the range based on the data with safety limits
            padding = 20.0  # Increased padding
            min_vals = np.min(pixels, axis=0)
            max_vals = np.max(pixels, axis=0)
            
            # Print the range to help debug
            print(f"Data range - Min: {min_vals}, Max: {max_vals}")
            
            axis_ranges = {
                'xaxis': {'range': [max(0, min_vals[0] - padding), min(180, max_vals[0] + padding)]},
                'yaxis': {'range': [max(0, min_vals[1] - padding), min(255, max_vals[1] + padding)]},
                'zaxis': {'range': [max(0, min_vals[2] - padding), min(255, max_vals[2] + padding)]}
            }
    
    # Update layout
    fig.update_layout(
        title=f'Image Color Distribution in {color_space} Space',
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            aspectmode='cube',
            **axis_ranges
        ),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40)  # Reduce margins
    )
    
    # Show the plot in the browser
    print("Opening plot in browser...")
    fig.show()

def visualize_image_color_clusters(image_path, color_space='RGB', n_clusters=5, sample_ratio=0.1, to_scale=False):
    """
    Visualize the color clusters of an image in 3D color space.
    
    Args:
        image_path (str): Path to the image
        color_space (str): Color space to use ('RGB' or 'HSV')
        n_clusters (int): Number of color clusters to identify
        sample_ratio (float): Ratio of pixels to sample
        to_scale (bool): Whether to show the full color space range
    """
    from sklearn.cluster import KMeans
    
    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    print(f"Image size: {img.shape}")
    
    # Convert to the specified color space
    img = assert_input_colour_space(img, color_space)
    
    # Reshape image to a list of pixels and ensure float type
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    print(f"Total pixels: {len(pixels)}")
    
    # Sample pixels to reduce computation
    if sample_ratio < 1.0:
        np.random.seed(42)  # For reproducibility
        sample_size = max(1000, int(len(pixels) * sample_ratio))  # Ensure at least 1000 points
        indices = np.random.choice(len(pixels), size=sample_size, replace=False)
        pixels = pixels[indices]
    
    print(f"Sampled pixels: {len(pixels)}")
    print(f"Clustering into {n_clusters} clusters...")
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    
    # Count pixels per cluster
    counts = np.bincount(labels)
    percentages = counts / len(pixels) * 100
    
    # Create a 3D scatter plot with clusters
    trace_data = []
    
    # Plot each cluster
    for i in range(n_clusters):
        cluster_pixels = pixels[labels == i]
        
        print(f"Cluster {i+1}: {len(cluster_pixels)} pixels ({percentages[i]:.1f}%)")
        
        # Convert cluster center to RGB for color representation
        if color_space == 'RGB':
            center_rgb = centers[i]
        else:  # HSV
            center_rgb = hsv_to_rgb(centers[i])
        
        # Create scatter plot for this cluster
        trace_data.append(
            go.Scatter3d(
                x=cluster_pixels[:, 0],
                y=cluster_pixels[:, 1],
                z=cluster_pixels[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=rgb_to_hex(center_rgb),
                    opacity=0.7
                ),
                name=f'Cluster {i+1}: {percentages[i]:.1f}%'
            )
        )
        
        # Add cluster center
        trace_data.append(
            go.Scatter3d(
                x=[centers[i][0]],
                y=[centers[i][1]],
                z=[centers[i][2]],
                mode='markers',
                marker=dict(
                    size=12,  # Increased size
                    color=rgb_to_hex(center_rgb),
                    line=dict(color='black', width=2),
                    symbol='diamond'  # Changed symbol for better visibility
                ),
                name=f'Center {i+1}'
            )
        )
    
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
            # Automatically set the range based on the data with safety limits
            padding = 20.0  # Increased padding
            min_vals = np.min(pixels, axis=0)
            max_vals = np.max(pixels, axis=0)
            
            # Print the range to help debug
            print(f"Data range - Min: {min_vals}, Max: {max_vals}")
            
            axis_ranges = {
                'xaxis': {'range': [max(0, min_vals[0] - padding), min(255, max_vals[0] + padding)]},
                'yaxis': {'range': [max(0, min_vals[1] - padding), min(255, max_vals[1] + padding)]},
                'zaxis': {'range': [max(0, min_vals[2] - padding), min(255, max_vals[2] + padding)]}
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
            # Automatically set the range based on the data with safety limits
            padding = 20.0  # Increased padding
            min_vals = np.min(pixels, axis=0)
            max_vals = np.max(pixels, axis=0)
            
            # Print the range to help debug
            print(f"Data range - Min: {min_vals}, Max: {max_vals}")
            
            axis_ranges = {
                'xaxis': {'range': [max(0, min_vals[0] - padding), min(180, max_vals[0] + padding)]},
                'yaxis': {'range': [max(0, min_vals[1] - padding), min(255, max_vals[1] + padding)]},
                'zaxis': {'range': [max(0, min_vals[2] - padding), min(255, max_vals[2] + padding)]}
            }
    
    # Create the figure
    fig = go.Figure(data=trace_data)
    
    # Update layout
    fig.update_layout(
        title=f'Image Color Clusters in {color_space} Space',
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            aspectmode='cube',
            **axis_ranges
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40)  # Reduce margins
    )
    
    # Show the plot in the browser
    print("Opening plot in browser...")
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize the colors of an image in 3D color space')
    parser.add_argument('image_path', help='Path to the image')
    parser.add_argument('--colour-space', choices=['RGB', 'HSV'], default='RGB',
                      help='Color space to use (default: RGB)')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                      help='Ratio of pixels to sample (default: 0.1)')
    parser.add_argument('--to-scale', action='store_true',
                      help='Show the full color space range')
    parser.add_argument('--clusters', type=int, default=0,
                      help='Number of color clusters to visualize (default: 0 - no clustering)')
    
    args = parser.parse_args()
    
    try:
        if args.clusters > 0:
            visualize_image_color_clusters(
                args.image_path, 
                args.colour_space, 
                n_clusters=args.clusters,
                sample_ratio=args.sample_ratio,
                to_scale=args.to_scale
            )
        else:
            visualize_image_colors(
                args.image_path, 
                args.colour_space, 
                sample_ratio=args.sample_ratio,
                to_scale=args.to_scale
            )
        print(f"Successfully visualized colors of {args.image_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 