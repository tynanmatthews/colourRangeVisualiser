import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os
from range_from_swatch import assert_input_colour_space
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.cluster import DBSCAN

def calculate_window_averages(image, window_size=100, color_space='RGB'):
    """
    Divides an image into non-overlapping windows and calculates the average color for each window.
    
    Args:
        image: Input image (numpy array)
        window_size: Size of each square window (default: 100)
        color_space: Color space to use ('RGB' or 'HSV')
        
    Returns:
        windows_grid: 2D grid of average colors
        window_positions: 2D grid of window positions (x, y, width, height)
    """
    # Convert image to specified color space if needed
    img = assert_input_colour_space(image, color_space)
    
    height, width = img.shape[:2]
    
    # Calculate number of windows in each dimension
    num_windows_x = width // window_size
    num_windows_y = height // window_size
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Window grid: {num_windows_x}x{num_windows_y} ({num_windows_x * num_windows_y} windows)")
    
    # Initialize arrays to store average colors and window positions
    windows_grid = np.zeros((num_windows_y, num_windows_x, 3), dtype=np.float32)
    window_positions = np.zeros((num_windows_y, num_windows_x, 4), dtype=np.int32)
    
    # Process each window
    for y in range(num_windows_y):
        for x in range(num_windows_x):
            # Calculate window coordinates
            x1 = x * window_size
            y1 = y * window_size
            x2 = min(x1 + window_size, width)
            y2 = min(y1 + window_size, height)
            
            # Store window position
            window_positions[y, x] = [x1, y1, window_size, window_size]
            
            # Extract window
            window = img[y1:y2, x1:x2]
            
            # Calculate average color
            avg_color = np.mean(window, axis=(0, 1))
            windows_grid[y, x] = avg_color
    
    return windows_grid, window_positions

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
    return rgb_pixel[0, 0]

def filter_noise_from_distinct_windows(distinct_indices, z_threshold=2.0, k_neighbors=5):
    """
    Filter out noisy distinct windows by detecting spatial outliers.
    
    Args:
        distinct_indices: List of (y, x) indices of distinct windows
        z_threshold: Z-score threshold for identifying outliers
        k_neighbors: Number of nearest neighbors to consider
        
    Returns:
        filtered_indices: List of filtered distinct window indices
    """
    if len(distinct_indices) <= k_neighbors + 1:
        print("Not enough distinct windows to perform noise filtering.")
        return distinct_indices
    
    # Convert indices to coordinate array
    coords = np.array(distinct_indices)
    
    # Calculate pairwise distances between all distinct windows
    distances = squareform(pdist(coords))
    
    # For each window, calculate the average distance to k nearest neighbors
    avg_neighbor_distances = []
    
    for i in range(len(distinct_indices)):
        # Get distances to all other windows and sort
        window_distances = distances[i]
        # Exclude self (0 distance) and get k nearest
        nearest_k = np.sort(window_distances)[1:k_neighbors+1]
        avg_neighbor_distances.append(np.mean(nearest_k))
    
    # Convert to numpy array
    avg_neighbor_distances = np.array(avg_neighbor_distances)
    
    # Calculate z-scores to identify outliers
    z_scores = stats.zscore(avg_neighbor_distances)
    
    # Find indices of non-outlier windows
    non_outlier_indices = np.where(np.abs(z_scores) <= z_threshold)[0]
    
    # Filter the distinct indices
    filtered_indices = [distinct_indices[i] for i in non_outlier_indices]
    
    removed_count = len(distinct_indices) - len(filtered_indices)
    print(f"Noise reduction removed {removed_count} outlier windows out of {len(distinct_indices)}.")
    
    return filtered_indices

def visualize_window_averages(windows_grid, window_positions, original_img, color_space='RGB', 
                             threshold=None, save_path=None):
    """
    Visualizes the average color of each window using Plotly for interactive browser-based visualization.
    
    Args:
        windows_grid: 2D grid of average colors
        window_positions: 2D grid of window positions
        original_img: Original image for comparison
        color_space: Color space used ('RGB' or 'HSV')
        threshold: Optional threshold value for highlighting windows
        save_path: Optional path to save the visualization
    """
    num_windows_y, num_windows_x = windows_grid.shape[:2]
    
    # Convert original image to RGB for display
    orig_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original Image', 'Average Color per 100x100 Window'),
        specs=[[{'type': 'image'}, {'type': 'image'}]]
    )
    
    # Add original image
    fig.add_trace(
        go.Image(z=orig_img_rgb),
        row=1, col=1
    )
    
    # Create the grid visualization
    grid_img = np.zeros((num_windows_y * 100, num_windows_x * 100, 3), dtype=np.uint8)
    
    # Fill the grid with average colors
    for y in range(num_windows_y):
        for x in range(num_windows_x):
            y1, y2 = y * 100, (y + 1) * 100
            x1, x2 = x * 100, (x + 1) * 100
            
            avg_color = windows_grid[y, x]
            
            # Convert HSV to RGB for visualization if needed
            if color_space == 'HSV':
                # Create a single pixel with the HSV color
                avg_color_rgb = hsv_to_rgb(avg_color)
            else:
                avg_color_rgb = avg_color
            
            # Fill the corresponding area in the grid image
            grid_img[y1:y2, x1:x2] = avg_color_rgb.astype(np.uint8)
    
    # Add grid image
    fig.add_trace(
        go.Image(z=grid_img),
        row=1, col=2
    )
    
    # Add grid lines and labels as shapes and annotations
    for y in range(num_windows_y + 1):
        fig.add_shape(
            type="line",
            x0=0, y0=y * 100, x1=num_windows_x * 100, y1=y * 100,
            line=dict(color="Black", width=1),
            row=1, col=2
        )
    
    for x in range(num_windows_x + 1):
        fig.add_shape(
            type="line",
            x0=x * 100, y0=0, x1=x * 100, y1=num_windows_y * 100,
            line=dict(color="Black", width=1),
            row=1, col=2
        )
    
    # Add window labels
    for y in range(num_windows_y):
        for x in range(num_windows_x):
            fig.add_annotation(
                x=x * 100 + 50, y=y * 100 + 50,
                text=f"({x},{y})",
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="white",
                borderwidth=1,
                borderpad=2,
                row=1, col=2
            )
    
    # Update layout
    fig.update_layout(
        title_text="Window Color Analysis",
        height=800,
        width=1600,
        showlegend=False
    )
    
    # Fix aspect ratio
    fig.update_xaxes(showticklabels=False, visible=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, visible=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, visible=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, visible=False, row=1, col=2)
    
    # If threshold is provided, create a third figure with highlighted windows
    if threshold is not None:
        # Determine the threshold type and apply
        if isinstance(threshold, tuple) and len(threshold) == 6:
            # Range threshold: (min_r, min_g, min_b, max_r, max_g, max_b)
            min_vals = np.array(threshold[:3])
            max_vals = np.array(threshold[3:])
            
            highlighted = np.all((windows_grid >= min_vals) & (windows_grid <= max_vals), axis=2)
        else:
            # Intensity threshold: Calculate color intensity (average of channels)
            color_intensity = np.mean(windows_grid, axis=2)
            highlighted = color_intensity > threshold
        
        # Count highlighted windows
        highlight_count = np.sum(highlighted)
        print(f"Highlighted windows: {highlight_count}")
        
        # Create highlighted figure
        highlight_fig = go.Figure()
        
        # Add original image
        highlight_fig.add_trace(go.Image(z=orig_img_rgb))
        
        # Add rectangles for highlighted windows
        for y in range(num_windows_y):
            for x in range(num_windows_x):
                if highlighted[y, x]:
                    pos = window_positions[y, x]
                    
                    # Add rectangle shape
                    highlight_fig.add_shape(
                        type="rect",
                        x0=pos[0], y0=pos[1], 
                        x1=pos[0] + pos[2], y1=pos[1] + pos[3],
                        line=dict(color="lime", width=2),
                        fillcolor="rgba(0,255,0,0.1)"
                    )
                    
                    # Add label
                    highlight_fig.add_annotation(
                        x=pos[0] + pos[2]//2, 
                        y=pos[1] + pos[3]//2,
                        text=f"({x},{y})",
                        showarrow=False,
                        font=dict(color="white", size=10),
                        bgcolor="rgba(0,255,0,0.7)",
                        bordercolor="white",
                        borderwidth=1,
                        borderpad=2
                    )
        
        # Update layout
        highlight_fig.update_layout(
            title_text="Highlighted Windows (Potential Objects)",
            height=800,
            width=800,
            showlegend=False
        )
        
        # Fix aspect ratio
        highlight_fig.update_xaxes(showticklabels=False, visible=False)
        highlight_fig.update_yaxes(showticklabels=False, visible=False)
        
        # Show highlighted figure
        highlight_fig.show()
        
        # Save the highlighted image if a path is provided
        if save_path:
            base, ext = os.path.splitext(save_path)
            highlight_fig.write_html(f"{base}_highlighted.html")
            highlight_fig.write_image(f"{base}_highlighted{ext}")
    
    # Show the main figure
    fig.show()
    
    # Save the visualization if a path is provided
    if save_path:
        fig.write_html(f"{os.path.splitext(save_path)[0]}.html")
        fig.write_image(save_path)

def visualize_3d_color_distribution(windows_grid, color_space='RGB'):
    """
    Creates an interactive 3D visualization of window colors.
    
    Args:
        windows_grid: 2D grid of average colors
        color_space: Color space used ('RGB' or 'HSV')
    """
    # Reshape the grid to get all window colors
    window_colors = windows_grid.reshape(-1, 3)
    
    # Create coordinates for the grid
    num_windows_y, num_windows_x = windows_grid.shape[:2]
    grid_coords = []
    window_labels = []
    
    for y in range(num_windows_y):
        for x in range(num_windows_x):
            grid_coords.append((x, y))
            window_labels.append(f"Window ({x},{y})")
    
    # Convert colors to RGB for display
    if color_space == 'HSV':
        marker_colors = [rgb_to_hex(hsv_to_rgb(color)) for color in window_colors]
    else:
        marker_colors = [rgb_to_hex(color) for color in window_colors]
    
    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=window_colors[:, 0],
            y=window_colors[:, 1],
            z=window_colors[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=marker_colors,
                opacity=0.8
            ),
            text=[f"{window_labels[i]}<br>Color: {window_colors[i]}" for i in range(len(window_colors))],
            hoverinfo='text'
        )
    ])
    
    # Set axis labels based on color space
    if color_space == 'RGB':
        axis_labels = ['Red', 'Green', 'Blue']
        axis_range = [0, 255]
    else:  # HSV
        axis_labels = ['Hue', 'Saturation', 'Value']
        axis_range = [[0, 180], [0, 255], [0, 255]]
    
    # Update layout
    fig.update_layout(
        title=f'3D Color Distribution in {color_space} Space',
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            xaxis=dict(range=axis_range[0] if isinstance(axis_range[0], list) else axis_range),
            yaxis=dict(range=axis_range[1] if isinstance(axis_range[0], list) else axis_range),
            zaxis=dict(range=axis_range[2] if isinstance(axis_range[0], list) else axis_range),
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Show the plot
    fig.show()
    
    return fig

def analyze_window_statistics(windows_grid, color_space='RGB'):
    """
    Analyzes and prints statistics about the window colors.
    
    Args:
        windows_grid: 2D grid of average colors
        color_space: Color space used ('RGB' or 'HSV')
    
    Returns:
        statistics: Dictionary containing color statistics
    """
    # Flatten the grid to analyze all windows
    all_windows = windows_grid.reshape(-1, 3)
    
    # Calculate statistics
    mean_color = np.mean(all_windows, axis=0)
    std_color = np.std(all_windows, axis=0)
    min_color = np.min(all_windows, axis=0)
    max_color = np.max(all_windows, axis=0)
    
    # Define channel names based on color space
    if color_space == 'RGB':
        channels = ['Red', 'Green', 'Blue']
    else:  # HSV
        channels = ['Hue', 'Saturation', 'Value']
    
    # Print statistics
    print("\nWindow Color Statistics:")
    print("-----------------------")
    for i, channel in enumerate(channels):
        print(f"{channel}: Mean={mean_color[i]:.1f}, StdDev={std_color[i]:.1f}, Min={min_color[i]:.1f}, Max={max_color[i]:.1f}")
    
    # Calculate color variation (useful for identifying potential objects)
    color_variation = np.sum(std_color)
    print(f"\nOverall color variation: {color_variation:.1f}")
    
    if color_variation > 50:
        print("High color variation - Multiple distinct color regions detected")
    else:
        print("Low color variation - Relatively uniform color across the image")
    
    # Create a visualization of the statistics
    stats_fig = go.Figure()
    
    # Add bar chart for mean values with error bars for standard deviation
    stats_fig.add_trace(go.Bar(
        x=channels,
        y=mean_color,
        error_y=dict(
            type='data',
            array=std_color,
            visible=True
        ),
        name='Mean Values'
    ))
    
    # Add range indicators (min/max)
    stats_fig.add_trace(go.Scatter(
        x=channels,
        y=min_color,
        mode='markers',
        name='Min Values',
        marker=dict(size=10, symbol='triangle-down')
    ))
    
    stats_fig.add_trace(go.Scatter(
        x=channels,
        y=max_color,
        mode='markers',
        name='Max Values',
        marker=dict(size=10, symbol='triangle-up')
    ))
    
    # Update layout
    stats_fig.update_layout(
        title="Color Channel Statistics",
        xaxis_title="Color Channel",
        yaxis_title="Value",
        yaxis=dict(range=[0, 260]) if color_space == 'RGB' else dict(range=[0, 260]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Show the figure
    stats_fig.show()
    
    # Return statistics as a dictionary
    return {
        'mean_color': mean_color,
        'std_color': std_color,
        'min_color': min_color,
        'max_color': max_color,
        'color_variation': color_variation
    }

def identify_objects(windows_grid, distinct_indices, color_space='RGB', 
                     color_weight=0.7, spatial_weight=0.3, eps=1.5, min_samples=2):
    """
    Identifies objects by clustering distinct windows based on color and spatial proximity.
    
    Args:
        windows_grid: 2D grid of average colors
        distinct_indices: List of (y, x) indices of distinct windows
        color_space: Color space used ('RGB' or 'HSV')
        color_weight: Weight for color distance (0-1)
        spatial_weight: Weight for spatial distance (0-1)
        eps: Maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point
        
    Returns:
        object_labels: Array of cluster labels for each distinct window
        merged_distinct_indices: Distinct indices array with additional column for object label
    """
    if len(distinct_indices) < min_samples:
        print("Not enough distinct windows to identify objects.")
        return np.zeros(len(distinct_indices), dtype=int), np.array([])
    
    # Create feature matrix with normalized spatial and color features
    num_windows_y, num_windows_x, _ = windows_grid.shape
    max_spatial_dist = np.sqrt(num_windows_x**2 + num_windows_y**2)
    
    # Extract colors for the distinct windows
    colors = np.array([windows_grid[y, x] for y, x in distinct_indices])
    
    # Normalize color values based on color space
    if color_space == 'RGB':
        max_color_val = 255.0
    else:  # HSV
        # Different normalization for each channel in HSV
        max_color_val = np.array([180.0, 255.0, 255.0])
        colors = colors / max_color_val[:, np.newaxis].T
    
    # For RGB, normalize all channels together
    if color_space == 'RGB':
        colors = colors / max_color_val
    
    # Create a custom distance matrix that combines color and spatial distances
    n_distinct = len(distinct_indices)
    combined_dist_matrix = np.zeros((n_distinct, n_distinct))
    
    for i in range(n_distinct):
        for j in range(i+1, n_distinct):
            # Spatial distance (Euclidean)
            y1, x1 = distinct_indices[i]
            y2, x2 = distinct_indices[j]
            spatial_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) / max_spatial_dist
            
            # Color distance (Euclidean in normalized color space)
            color_dist = np.linalg.norm(colors[i] - colors[j])
            
            # Weighted combined distance
            combined_dist = color_weight * color_dist + spatial_weight * spatial_dist
            combined_dist_matrix[i, j] = combined_dist
            combined_dist_matrix[j, i] = combined_dist
    
    # Apply DBSCAN clustering using the precomputed distance matrix
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    object_labels = db.fit_predict(combined_dist_matrix)
    
    # Count the number of objects found (excluding noise with label -1)
    n_objects = len(set(object_labels[object_labels >= 0]))
    print(f"Identified {n_objects} distinct objects in the image.")
    
    # Merge distinct indices with object labels
    merged_distinct_indices = np.column_stack((distinct_indices, object_labels))
    
    return object_labels, merged_distinct_indices

def visualize_object_clusters(windows_grid, merged_distinct_indices, original_img, color_space='RGB', save_path=None):
    """
    Visualizes the identified object clusters.
    
    Args:
        windows_grid: 2D grid of average colors
        merged_distinct_indices: Distinct indices array with object labels
        original_img: Original image
        color_space: Color space used ('RGB' or 'HSV')
        save_path: Optional path to save the visualization
    """
    # Extract distinct indices and object labels
    distinct_indices = merged_distinct_indices[:, :2].astype(int)
    object_labels = merged_distinct_indices[:, 2].astype(int)
    
    # Convert original image to RGB for display
    orig_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Create a figure to visualize the objects
    object_fig = go.Figure()
    object_fig.add_trace(go.Image(z=orig_img_rgb))
    
    # Get num_windows_x and num_windows_y from windows_grid shape
    num_windows_y, num_windows_x, _ = windows_grid.shape
    window_size = 100  # Default window size
    
    # Define a colormap for different object labels
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # Get unique object labels (excluding noise with label -1)
    unique_objects = sorted(set(object_labels[object_labels >= 0]))
    n_objects = len(unique_objects)
    
    # Create a colormap with distinct colors
    if n_objects > 0:
        colormap = cm.get_cmap('tab10', max(10, n_objects))
        object_colors = {label: mcolors.rgb2hex(colormap(i)[:3]) for i, label in enumerate(unique_objects)}
        
        # Add noise in grey if there's any
        if -1 in object_labels:
            object_colors[-1] = '#808080'  # Grey for noise
    else:
        object_colors = {-1: '#808080'}  # Grey for noise if only noise exists
    
    # Create window positions
    window_positions = np.zeros((num_windows_y, num_windows_x, 4), dtype=np.int32)
    for y in range(num_windows_y):
        for x in range(num_windows_x):
            x1 = x * window_size
            y1 = y * window_size
            window_positions[y, x] = [x1, y1, window_size, window_size]
    
    # Add rectangles and labels for each object
    for label in sorted(set(object_labels)):
        label_indices = [i for i, obj_label in enumerate(object_labels) if obj_label == label]
        if label == -1:
            rect_color = "gray"  # Noise points
            alpha = 0.2
            title = "Noise"
        else:
            rect_color = object_colors[label]
            alpha = 0.4
            title = f"Object {label}"
        
        # Create a polygon outline to connect all windows in this object
        if label >= 0 and len(label_indices) > 2:  # Only for actual objects with enough points
            # Extract window coordinates (center points)
            points_y = [distinct_indices[i][0] * window_size + window_size//2 for i in label_indices]
            points_x = [distinct_indices[i][1] * window_size + window_size//2 for i in label_indices]
            
            # Compute the convex hull of these points
            if len(points_y) >= 3:  # Need at least 3 points for a convex hull
                try:
                    from scipy.spatial import ConvexHull
                    points = np.column_stack([points_x, points_y])
                    hull = ConvexHull(points)
                    hull_points_x = [points[hull.vertices, 0]]
                    hull_points_y = [points[hull.vertices, 1]]
                    
                    # Add the hull outline
                    object_fig.add_trace(go.Scatter(
                        x=np.append(hull_points_x, hull_points_x[0][0]),
                        y=np.append(hull_points_y, hull_points_y[0][0]),
                        mode='lines',
                        line=dict(color=rect_color, width=3),
                        name=title,
                        showlegend=True
                    ))
                except Exception as e:
                    print(f"Could not compute convex hull for object {label}: {str(e)}")
        
        # Add individual window rectangles
        for i in label_indices:
            y, x = distinct_indices[i]
            pos = window_positions[y, x]
            
            # Add rectangle shape
            object_fig.add_shape(
                type="rect",
                x0=pos[0], y0=pos[1], 
                x1=pos[0] + pos[2], y1=pos[1] + pos[3],
                line=dict(color=rect_color, width=2),
                fillcolor="plum",
                name=title,
                showlegend=False
            )
            
            # Add label for the first window of each object
            if i == label_indices[0]:
                object_fig.add_annotation(
                    x=pos[0] + pos[2]//2, 
                    y=pos[1] + pos[3]//2,
                    text=title,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=rect_color,
                    font=dict(color="white", size=12),
                    bgcolor="plum",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4
                )
    
    # Update layout
    object_fig.update_layout(
        title_text="Identified Objects",
        height=800,
        width=1000,
        showlegend=True,
        legend=dict(
            title="Objects",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Fix aspect ratio
    object_fig.update_xaxes(showticklabels=False, visible=False)
    object_fig.update_yaxes(showticklabels=False, visible=False)
    
    # Show the figure
    object_fig.show()
    
    # Save the visualization if a path is provided
    if save_path:
        base, ext = os.path.splitext(save_path)
        object_fig.write_html(f"{base}_objects.html")
        object_fig.write_image(f"{base}_objects{ext}")
    
    return object_fig

def find_distinct_windows(windows_grid, color_space='RGB', std_threshold=1.5, reduce_noise=False, identify_objects_flag=False):
    """
    Finds windows with colors that stand out from the overall image and identifies objects.
    
    Args:
        windows_grid: 2D grid of average colors
        color_space: Color space used ('RGB' or 'HSV')
        std_threshold: How many standard deviations from mean to consider distinct
        reduce_noise: Whether to filter out isolated distinct windows
        identify_objects_flag: Whether to identify and visualize objects
        
    Returns:
        distinct_indices: List of (y, x) indices of distinct windows
    """
    # Flatten the grid for overall statistics
    all_windows = windows_grid.reshape(-1, 3)
    
    # Calculate mean and standard deviation
    mean_color = np.mean(all_windows, axis=0)
    std_color = np.std(all_windows, axis=0)
    
    # Reshape windows_grid to original dimensions
    num_windows_y, num_windows_x, _ = windows_grid.shape
    
    # Find windows with colors significantly different from the mean
    distinct_indices = []
    std_diff_values = []
    
    for y in range(num_windows_y):
        for x in range(num_windows_x):
            window_color = windows_grid[y, x]
            
            # Calculate how many standard deviations away from the mean
            std_diff = np.abs(window_color - mean_color) / (std_color + 1e-10)  # Avoid division by zero
            max_std_diff = np.max(std_diff)
            std_diff_values.append((y, x, max_std_diff))
            
            # If any channel exceeds the threshold, consider it distinct
            if np.any(std_diff > std_threshold):
                distinct_indices.append((y, x))
    
    # Filter noise if requested
    original_count = len(distinct_indices)
    if reduce_noise and original_count > 0:
        print(f"\nApplying noise reduction to {original_count} distinct windows...")
        distinct_indices = filter_noise_from_distinct_windows(distinct_indices)
    
    # Print the distinct windows
    if distinct_indices:
        if color_space == 'RGB':
            channels = ['R', 'G', 'B']
        else:  # HSV
            channels = ['H', 'S', 'V']
            
        print(f"\nFound {len(distinct_indices)} distinct windows:")
        for y, x in distinct_indices:
            window_color = windows_grid[y, x]
            color_str = ", ".join(f"{channels[i]}={window_color[i]:.1f}" for i in range(3))
            print(f"Window ({x},{y}) - {color_str}")
            
        # Create a visualization of distinct windows
        # Create 3D scatter plot
        distinct_fig = go.Figure()
        
        # Add all windows as small points
        all_points_x = []
        all_points_y = []
        all_text = []
        
        for y in range(num_windows_y):
            for x in range(num_windows_x):
                all_points_x.append(x)
                all_points_y.append(y)
                all_text.append(f"Window ({x},{y})<br>Color: {windows_grid[y, x]}")
        
        distinct_fig.add_trace(go.Scatter(
            x=all_points_x,
            y=all_points_y,
            mode='markers',
            marker=dict(
                size=10,
                color='lightgrey',
                opacity=0.5
            ),
            text=all_text,
            hoverinfo='text',
            name='Regular Windows'
        ))
        
        # Add distinct windows as larger points
        distinct_x = [x for y, x in distinct_indices]
        distinct_y = [y for y, x in distinct_indices]
        distinct_text = [f"DISTINCT: Window ({x},{y})<br>Color: {windows_grid[y, x]}" for y, x in distinct_indices]
        
        # Convert window colors to hex for marker colors
        if color_space == 'HSV':
            distinct_colors = [rgb_to_hex(hsv_to_rgb(windows_grid[y, x])) for y, x in distinct_indices]
        else:
            distinct_colors = [rgb_to_hex(windows_grid[y, x]) for y, x in distinct_indices]
        
        distinct_fig.add_trace(go.Scatter(
            x=distinct_x,
            y=distinct_y,
            mode='markers',
            marker=dict(
                size=15,
                color=distinct_colors,
                line=dict(width=2, color='black')
            ),
            text=distinct_text,
            hoverinfo='text',
            name='Distinct Windows'
        ))
        
        # If noise reduction was applied, also show removed windows
        if reduce_noise and original_count > len(distinct_indices):
            # Find windows that were removed
            current_set = set(tuple(idx) for idx in distinct_indices)
            original_set = set()
            for y, x, _ in std_diff_values:
                window_color = windows_grid[y, x]
                std_diff = np.abs(window_color - mean_color) / (std_color + 1e-10)
                if np.any(std_diff > std_threshold):
                    original_set.add((y, x))
            
            removed_set = original_set - current_set
            
            # Add removed windows as crosses
            removed_x = [x for y, x in removed_set]
            removed_y = [y for y, x in removed_set]
            removed_text = [f"FILTERED OUT: Window ({x},{y})<br>Color: {windows_grid[y, x]}" for y, x in removed_set]
            
            # Convert window colors to hex
            if color_space == 'HSV':
                removed_colors = [rgb_to_hex(hsv_to_rgb(windows_grid[y, x])) for y, x in removed_set]
            else:
                removed_colors = [rgb_to_hex(windows_grid[y, x]) for y, x in removed_set]
            
            distinct_fig.add_trace(go.Scatter(
                x=removed_x,
                y=removed_y,
                mode='markers',
                marker=dict(
                    size=12,
                    color=removed_colors,
                    symbol='x',
                    line=dict(width=2, color='red')
                ),
                text=removed_text,
                hoverinfo='text',
                name='Filtered Out (Noise)'
            ))
        
        # Update layout
        distinct_fig.update_layout(
            title="Distinct Windows Detection",
            xaxis_title="X Window Index",
            yaxis_title="Y Window Index",
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                range=[-0.5, num_windows_x - 0.5]
            ),
            yaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                range=[num_windows_y - 0.5, -0.5]  # Inverted for image coordinates
            ),
            showlegend=True
        )
        
        # Show the figure
        distinct_fig.show()
        
        # Identify objects if requested
        if identify_objects_flag and len(distinct_indices) >= 2:
            object_labels, merged_indices = identify_objects(windows_grid, distinct_indices, color_space)
            return distinct_indices, object_labels, merged_indices
        
    else:
        print("\nNo distinctly colored windows found.")
    
    return distinct_indices, None, None

def process_image(image_path, color_space='RGB', window_size=100, threshold=None, 
                 find_distinct=False, std_threshold=1.5, reduce_noise=False, 
                 identify_objects_flag=False, color_weight=0.7, save_path=None):
    """
    Process an image to visualize average colors in windows and identify objects.
    
    Args:
        image_path: Path to the input image
        color_space: Color space to use ('RGB' or 'HSV')
        window_size: Size of each square window
        threshold: Optional threshold value for highlighting
        find_distinct: Whether to find windows with distinct colors
        std_threshold: How many standard deviations from mean to consider distinct
        reduce_noise: Whether to filter out isolated distinct windows
        identify_objects_flag: Whether to identify and visualize objects
        color_weight: Weight for color distance in object identification (0-1)
        save_path: Optional path to save the visualization
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    print(f"Processing image: {image_path}")
    print(f"Color space: {color_space}")
    
    # Calculate average colors for windows
    windows_grid, window_positions = calculate_window_averages(img, window_size, color_space)
    
    # Analyze window statistics
    statistics = analyze_window_statistics(windows_grid, color_space)
    
    # Create 3D color distribution visualization
    visualize_3d_color_distribution(windows_grid, color_space)
    
    # Find windows with distinct colors if requested
    object_labels = None
    merged_indices = None
    if find_distinct:
        distinct_indices, object_labels, merged_indices = find_distinct_windows(
            windows_grid, color_space, std_threshold, reduce_noise, identify_objects_flag)
        
        # Visualize identified objects if requested
        if identify_objects_flag and merged_indices is not None:
            visualize_object_clusters(windows_grid, merged_indices, img, color_space, save_path)
    
    # Visualize the results
    # visualize_window_averages(windows_grid, window_positions, img, color_space, threshold, save_path)
    
    return windows_grid, window_positions, statistics, object_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze average colors in image windows')
    parser.add_argument('image_path', help='Path to the image')
    parser.add_argument('--colour-space', choices=['RGB', 'HSV'], default='RGB',
                      help='Color space to use (default: RGB)')
    parser.add_argument('--window-size', type=int, default=100,
                      help='Size of each window in pixels (default: 100)')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Intensity threshold for highlighting windows (0-255)')
    parser.add_argument('--find-distinct', action='store_true',
                      help='Find windows with colors that stand out')
    parser.add_argument('--std-threshold', type=float, default=1.5,
                      help='Standard deviation threshold for distinct windows (default: 1.5)')
    parser.add_argument('--reduce-noise', action='store_true',
                      help='Filter out isolated distinct windows to reduce noise')
    parser.add_argument('--identify-objects', action='store_true',
                      help='Identify distinct objects in the image')
    parser.add_argument('--color-weight', type=float, default=0.7,
                      help='Weight for color distance in object identification (0-1, default: 0.7)')
    parser.add_argument('--save', type=str, default=None,
                      help='Path to save the visualization')
    
    args = parser.parse_args()
    
    try:
        process_image(
            args.image_path,
            args.colour_space,
            args.window_size,
            args.threshold,
            args.find_distinct,
            args.std_threshold,
            args.reduce_noise,
            args.identify_objects,
            args.color_weight,
            args.save
        )
        print("Analysis complete!")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 