import argparse
from visualize_range import process_and_visualize_swatch

def main():
    parser = argparse.ArgumentParser(description='Process a swatch image and visualize its color range')
    parser.add_argument('image_path', help='Path to the swatch image')
    parser.add_argument('--colour-space', choices=['RGB', 'HSV'], default='RGB',
                      help='Color space to use (default: RGB)')
    parser.add_argument('--discrim-bands', type=int, default=3,
                      help='Number of k-means clusters (default: 3)')
    parser.add_argument('--retention', type=float, default=50.0,
                      help='Percentage of points to retain (default: 50.0)')
    
    args = parser.parse_args()
    
    config = {
        'colour_space': args.colour_space,
        'discrim_bands': args.discrim_bands,
        'retention': args.retention
    }
    
    try:
        color_range = process_and_visualize_swatch(args.image_path, config)
        print(f"Successfully processed swatch. Color range: {color_range}")
    except Exception as e:
        print(f"Error processing swatch: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()