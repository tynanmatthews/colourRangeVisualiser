import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import argparse
import os
import base64
import io
from PIL import Image
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from window_color_analyzer
from visualiser.window_color_analyzer import (
    calculate_window_averages,
    find_distinct_windows,
    visualize_3d_color_distribution,
    analyze_window_statistics
)

# Import functions from colourpicker
try:
    from colourpicker.range_from_swatch import extractColourRangeFromSwatch
except ImportError:
    # Create a simplified version if the full module isn't available
    def extractColourRangeFromSwatch(colors, color_space='RGB'):
        """
        Simplified version of extractColourRangeFromSwatch that works with 
        a list of colors rather than a swatch image.
        """
        if not colors:
            return None
            
        # Convert colors to a numpy array
        colors_array = np.array(colors)
        
        # Determine min and max values for each channel
        min_vals = np.min(colors_array, axis=0)
        max_vals = np.max(colors_array, axis=0)
        
        # Return range as [min, max]
        return [min_vals, max_vals]

def parse_image(contents):
    """
    Decode the base64 image content
    """
    if contents is None:
        return None
        
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Convert to opencv format
    img_array = np.frombuffer(decoded, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    return img

def assert_input_colour_space(image, space):
    """
    Convert image to the specified color space.
    """
    if space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image

def create_app():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Color Range Generator for Image Objects", className="text-center my-4"),
                html.P(
                    "Upload an image, analyze it to find distinct color windows, "
                    "select windows of interest, and generate a color range for use with the colourpicker algorithm.",
                    className="lead text-center"
                ),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Step 1: Upload Image & Settings"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-image',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select an Image')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                        html.Div(id='output-image-upload'),
                        
                        html.Hr(),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Color Space:"),
                                dcc.RadioItems(
                                    id='color-space',
                                    options=[
                                        {'label': 'RGB', 'value': 'RGB'},
                                        {'label': 'HSV', 'value': 'HSV'}
                                    ],
                                    value='RGB',
                                    inline=True
                                ),
                            ], width=6),
                            
                            dbc.Col([
                                html.Label("Window Size:"),
                                dcc.Input(
                                    id='window-size',
                                    type='number',
                                    value=100,
                                    min=50,
                                    max=200,
                                    step=10
                                ),
                            ], width=6),
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Standard Deviation Threshold:"),
                                dcc.Slider(
                                    id='std-threshold',
                                    min=0.5,
                                    max=3.0,
                                    step=0.1,
                                    value=1.5,
                                    marks={i: str(i) for i in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]}
                                ),
                            ], width=12),
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Checkbox(
                                    id='reduce-noise',
                                    label="Reduce Noise",
                                    value=True
                                ),
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    "Analyze Image",
                                    id="analyze-button",
                                    color="primary",
                                    className="mt-3"
                                ),
                            ], width=6),
                        ]),
                    ]),
                ], className="mb-4"),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Step 2: Select Distinct Windows"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-distinct-windows",
                            type="circle",
                            children=html.Div(id='distinct-windows-output')
                        ),
                        
                        html.Div(id='selection-instructions', style={'display': 'none'}, children=[
                            html.P("Click on the distinct windows you want to include in your color range.", className="mt-3"),
                            html.P("Selected windows will be used to generate a color range for the colourpicker algorithm."),
                        ]),
                        
                        dcc.Graph(id='distinct-windows-plot'),
                        
                        html.Div(id='selected-windows-info', className="mt-3"),
                        
                        dbc.Button(
                            "Generate Color Range",
                            id="generate-range-button",
                            color="success",
                            className="mt-3",
                            style={'display': 'none'}
                        ),
                    ]),
                ], className="mb-4"),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Step 3: Generated Color Range"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-color-range",
                            type="circle",
                            children=html.Div(id='color-range-output')
                        ),
                        
                        html.Div(id='color-range-display'),
                        
                        html.Div(id='color-range-copy-area', className="mt-3"),
                        
                        dbc.Button(
                            "Apply Range to Image",
                            id="apply-range-button",
                            color="info",
                            className="mt-3",
                            style={'display': 'none'}
                        ),
                    ]),
                ], className="mb-4"),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Step 4: Results"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-results",
                            type="circle",
                            children=html.Div(id='results-output')
                        ),
                        
                        dcc.Graph(id='results-plot'),
                    ]),
                ]),
            ], width=12),
        ]),
        
        # Store components to hold the state
        dcc.Store(id='image-store'),
        dcc.Store(id='windows-grid-store'),
        dcc.Store(id='distinct-indices-store'),
        dcc.Store(id='selected-windows-store'),
        dcc.Store(id='color-range-store'),
    ], fluid=True)
    
    @app.callback(
        [Output('output-image-upload', 'children'),
         Output('image-store', 'data')],
        [Input('upload-image', 'contents')],
        [State('upload-image', 'filename')]
    )
    def update_output(contents, filename):
        if contents is None:
            return no_update, no_update
            
        img = parse_image(contents)
        if img is None:
            return html.Div(['Invalid image file']), no_update
            
        # Store the image in base64 format
        _, buffer = cv2.imencode('.png', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Show a preview of the image
        return html.Div([
            html.Img(src=contents, style={'maxWidth': '100%', 'maxHeight': '300px'}),
            html.Div(f'Uploaded: {filename}')
        ]), img_str
    
    @app.callback(
        [Output('distinct-windows-output', 'children'),
         Output('distinct-windows-plot', 'figure'),
         Output('windows-grid-store', 'data'),
         Output('distinct-indices-store', 'data'),
         Output('selection-instructions', 'style'),
         Output('generate-range-button', 'style')],
        [Input('analyze-button', 'n_clicks')],
        [State('image-store', 'data'),
         State('color-space', 'value'),
         State('window-size', 'value'),
         State('std-threshold', 'value'),
         State('reduce-noise', 'value')]
    )
    def analyze_image(n_clicks, img_str, color_space, window_size, std_threshold, reduce_noise):
        if n_clicks is None or img_str is None:
            return no_update, no_update, no_update, no_update, no_update, no_update
            
        # Decode the image
        img_data = base64.b64decode(img_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Calculate window averages
        windows_grid, window_positions = calculate_window_averages(img, window_size, color_space)
        
        # Find distinct windows
        distinct_indices, _, _ = find_distinct_windows(
            windows_grid, color_space, std_threshold, reduce_noise, False)
        
        if not distinct_indices:
            return html.Div(["No distinct windows found. Try adjusting the threshold."]), no_update, no_update, no_update, {'display': 'none'}, {'display': 'none'}
        
        # Create a visualization of distinct windows
        fig = go.Figure()
        
        # Add all windows as small points
        num_windows_y, num_windows_x, _ = windows_grid.shape
        all_points_x = []
        all_points_y = []
        all_text = []
        
        for y in range(num_windows_y):
            for x in range(num_windows_x):
                all_points_x.append(x)
                all_points_y.append(y)
                all_text.append(f"Window ({x},{y})<br>Color: {windows_grid[y, x]}")
        
        fig.add_trace(go.Scatter(
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
            distinct_colors = []
            for y, x in distinct_indices:
                hsv = windows_grid[y, x]
                # Convert HSV to RGB for display
                hsv_pixel = np.array([[[hsv[0], hsv[1], hsv[2]]]], dtype=np.uint8)
                rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)
                r, g, b = rgb_pixel[0, 0]
                distinct_colors.append(f'rgb({r}, {g}, {b})')
        else:
            distinct_colors = []
            for y, x in distinct_indices:
                r, g, b = windows_grid[y, x]
                distinct_colors.append(f'rgb({int(r)}, {int(g)}, {int(b)})')
        
        fig.add_trace(go.Scatter(
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
        
        # Update layout
        fig.update_layout(
            title="Click to Select Distinct Windows",
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
            showlegend=True,
            clickmode='event+select'
        )
        
        # Convert numpy arrays to lists for JSON serialization
        windows_grid_list = windows_grid.tolist()
        distinct_indices_list = [list(idx) for idx in distinct_indices]
        
        return html.Div([
            html.H5(f"Found {len(distinct_indices)} distinct windows"),
            html.P("Click on the windows in the plot below to select them for your color range.")
        ]), fig, windows_grid_list, distinct_indices_list, {'display': 'block'}, {'display': 'block'}
    
    @app.callback(
        [Output('selected-windows-info', 'children'),
         Output('selected-windows-store', 'data')],
        [Input('distinct-windows-plot', 'clickData')],
        [State('windows-grid-store', 'data'),
         State('distinct-indices-store', 'data'),
         State('selected-windows-store', 'data'),
         State('color-space', 'value')]
    )
    def update_selected_windows(click_data, windows_grid, distinct_indices, selected_windows, color_space):
        if click_data is None or windows_grid is None or distinct_indices is None:
            return no_update, []
            
        # Convert to numpy arrays
        windows_grid = np.array(windows_grid)
        distinct_indices = [tuple(idx) for idx in distinct_indices]
        
        # Get the clicked point
        point_idx = click_data['points'][0]['pointIndex']
        x = click_data['points'][0]['x']
        y = click_data['points'][0]['y']
        
        # Check if the clicked point is a distinct window
        if (y, x) not in distinct_indices:
            return no_update, selected_windows or []
            
        # Initialize selected windows if not exist
        if selected_windows is None:
            selected_windows = []
        else:
            selected_windows = list(selected_windows)
        
        # Toggle selection
        window_key = [y, x]
        if window_key in selected_windows:
            selected_windows.remove(window_key)
        else:
            selected_windows.append(window_key)
        
        # Display information about selected windows
        if not selected_windows:
            return html.Div(["No windows selected"]), []
            
        # Display color information
        window_info = []
        for y, x in selected_windows:
            color = windows_grid[y, x]
            if color_space == 'RGB':
                color_text = f"R={color[0]:.1f}, G={color[1]:.1f}, B={color[2]:.1f}"
            else:  # HSV
                color_text = f"H={color[0]:.1f}, S={color[1]:.1f}, V={color[2]:.1f}"
                
            window_info.append(html.Div([
                f"Window ({x},{y}): {color_text}",
                html.Div(style={
                    'display': 'inline-block',
                    'width': '20px',
                    'height': '20px',
                    'backgroundColor': f'rgb({int(color[0])}, {int(color[1])}, {int(color[2])})' if color_space == 'RGB' else None,
                    'marginLeft': '10px',
                    'border': '1px solid black'
                })
            ]))
        
        return html.Div([
            html.H6(f"Selected Windows: {len(selected_windows)}"),
            html.Div(window_info)
        ]), selected_windows
    
    @app.callback(
        [Output('color-range-output', 'children'),
         Output('color-range-display', 'children'),
         Output('color-range-copy-area', 'children'),
         Output('color-range-store', 'data'),
         Output('apply-range-button', 'style')],
        [Input('generate-range-button', 'n_clicks')],
        [State('windows-grid-store', 'data'),
         State('selected-windows-store', 'data'),
         State('color-space', 'value')]
    )
    def generate_color_range(n_clicks, windows_grid, selected_windows, color_space):
        if n_clicks is None or not selected_windows or windows_grid is None:
            return no_update, no_update, no_update, no_update, no_update
            
        # Convert to numpy arrays
        windows_grid = np.array(windows_grid)
        
        # Extract colors from selected windows
        colors = [windows_grid[y, x] for y, x in selected_windows]
        
        # Generate range
        config = {
            'colour_space': color_space,
            'discrim_bands': 3,  # Number of clusters
            'retention': 100  # Retention percentage
        }
        
        try:
            # Create a simplified range computation if extractColourRangeFromSwatch isn't fully available
            min_vals = np.min(colors, axis=0)
            max_vals = np.max(colors, axis=0)
            color_range = [min_vals.tolist(), max_vals.tolist()]
            
            # Format for display
            if color_space == 'RGB':
                min_display = f"R={color_range[0][0]:.1f}, G={color_range[0][1]:.1f}, B={color_range[0][2]:.1f}"
                max_display = f"R={color_range[1][0]:.1f}, G={color_range[1][1]:.1f}, B={color_range[1][2]:.1f}"
            else:  # HSV
                min_display = f"H={color_range[0][0]:.1f}, S={color_range[0][1]:.1f}, V={color_range[0][2]:.1f}"
                max_display = f"H={color_range[1][0]:.1f}, S={color_range[1][1]:.1f}, V={color_range[1][2]:.1f}"
            
            # Create color preview
            color_preview = html.Div([
                html.H6("Color Range Preview:"),
                html.Div([
                    html.Div([
                        html.Div("Min:", style={'fontWeight': 'bold'}),
                        html.Div(min_display),
                        html.Div(style={
                            'width': '50px',
                            'height': '50px',
                            'backgroundColor': f'rgb({int(color_range[0][0])}, {int(color_range[0][1])}, {int(color_range[0][2])})' if color_space == 'RGB' else None,
                            'border': '1px solid black',
                            'margin': '5px'
                        })
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
                    html.Div([
                        html.Div("Max:", style={'fontWeight': 'bold'}),
                        html.Div(max_display),
                        html.Div(style={
                            'width': '50px',
                            'height': '50px',
                            'backgroundColor': f'rgb({int(color_range[1][0])}, {int(color_range[1][1])}, {int(color_range[1][2])})' if color_space == 'RGB' else None,
                            'border': '1px solid black',
                            'margin': '5px'
                        })
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'})
                ])
            ])
            
            # Create copyable text
            copy_text = f"Color Range ({color_space}):\n"
            copy_text += f"Min: [{', '.join([f'{val:.1f}' for val in color_range[0]])}]\n"
            copy_text += f"Max: [{', '.join([f'{val:.1f}' for val in color_range[1]])}]"
            
            copy_area = html.Div([
                html.H6("Copy this range for use with colourpicker:"),
                dcc.Textarea(
                    value=copy_text,
                    style={'width': '100%', 'height': '100px'}
                )
            ])
            
            return html.Div([
                html.H5("Color Range Generated Successfully"),
                html.P(f"Based on {len(selected_windows)} selected windows")
            ]), color_preview, copy_area, color_range, {'display': 'block'}
            
        except Exception as e:
            return html.Div([
                html.H5("Error Generating Color Range"),
                html.P(str(e))
            ]), no_update, no_update, no_update, {'display': 'none'}
    
    @app.callback(
        [Output('results-output', 'children'),
         Output('results-plot', 'figure')],
        [Input('apply-range-button', 'n_clicks')],
        [State('image-store', 'data'),
         State('color-range-store', 'data'),
         State('color-space', 'value'),
         State('window-size', 'value')]
    )
    def apply_color_range(n_clicks, img_str, color_range, color_space, window_size):
        if n_clicks is None or img_str is None or color_range is None:
            return no_update, no_update
            
        # Decode the image
        img_data = base64.b64decode(img_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Convert image to specified color space
        img_color = assert_input_colour_space(img, color_space)
        
        # Create mask using the color range
        min_vals = np.array(color_range[0], dtype=np.uint8)
        max_vals = np.array(color_range[1], dtype=np.uint8)
        mask = cv2.inRange(img_color, min_vals, max_vals)
        
        # Apply mask to image
        result = cv2.bitwise_and(img, img, mask=mask)
        
        # Count pixels in mask
        mask_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        percentage = (mask_pixels / total_pixels) * 100
        
        # Convert results to RGB for display
        orig_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original Image', 'Detected Objects')
        )
        
        # Add original image
        fig.add_trace(
            go.Image(z=orig_img_rgb),
            row=1, col=1
        )
        
        # Add result image
        fig.add_trace(
            go.Image(z=result_rgb),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Objects Detected using {color_space} Color Range",
            height=600,
            width=1200
        )
        
        return html.Div([
            html.H5("Object Detection Results"),
            html.P(f"Detected {mask_pixels} pixels ({percentage:.2f}% of image) matching the color range.")
        ]), fig
    
    return app

def main():
    parser = argparse.ArgumentParser(description='Color Range UI for Object Detection')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    app = create_app()
    app.run(debug=args.debug, port=args.port)

if __name__ == '__main__':
    main() 