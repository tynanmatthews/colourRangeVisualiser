import cv2 as cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_image_from_file(file_path):
    """
    Read an image from a local file.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: The image as a numpy array
    """
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read image from {file_path}")
    return img

def assert_input_colour_space(img, colour_space):
    """
    Convert image to specified color space if needed.
    
    Args:
        img (numpy.ndarray): Input image
        colour_space (str): Target color space ('RGB' or 'HSV')
        
    Returns:
        numpy.ndarray: Image in the specified color space
    """
    if colour_space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colour_space == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported color space: {colour_space}")

def extract_colour_range_from_swatch(file_path, config):
    """
    Extract color range from a swatch image using k-means clustering.
    
    Args:
        file_path (str): Path to the swatch image
        config (dict): Configuration dictionary containing:
            - colour_space: 'RGB' or 'HSV'
            - discrim_bands: number of k-means clusters
            - retention: percentage of points to retain
            
    Returns:
        list: Two points defining the color range [[min_c1, min_c2, min_c3], [max_c1, max_c2, max_c3]]
    """
    try:
        # Read image from file
        img = read_image_from_file(file_path)
        
        logger.info("Running range extraction")
        logger.info(f"Config: {config}")

        # Convert to specified color space
        img = assert_input_colour_space(img, config['colour_space'])

        # Reshape image for k-means
        channels = 3
        img_lin = img.reshape((-1, channels))
        Z = np.float32(img_lin)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = config['discrim_bands']
        ret, label, center = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        logger.info(f"Total labels: {len(label)}")

        # Count points per cluster
        clustered = np.zeros(K)
        for cluster in label:
            clustered[cluster] = clustered[cluster] + 1

        # Find the cluster with most points
        winner = np.argmax(clustered)
        refpoint = center[winner]
        
        logger.info(f"Majority in cluster:{winner} = {clustered[winner]}")
        logger.info(f"Reference centre = {refpoint}")

        # Measure point distances in majority cluster
        spread_point = np.empty((int(clustered[winner]), channels+1))
        spread_n = 0

        for i in range(len(label)):
            if label[i] == winner:
                spread_point[spread_n] = [np.linalg.norm(
                    img_lin[i]-refpoint), *img_lin[i]]
                spread_n += 1

        # Sort by distance and get retention percentage
        spread_point = spread_point[spread_point[:, 0].argsort()]
        top = int((config['retention'] / 100) * (len(spread_point)-1))
        logger.info(f"Retaining up to: {top}")

        # Get the range
        ranged = [(spread_point[0][1:4]), (spread_point[top][1:(channels+1)])]
        logger.info(f"Derived directed range: {ranged}")

        # Ensure min/max ordering
        for i in range(channels):
            if ranged[0][i] > ranged[1][i]:
                ranged[0][i], ranged[1][i] = ranged[1][i], ranged[0][i]

        logger.info(f"Final range: {ranged}")
        return ranged

    except Exception as e:
        logger.error(f"Error in color range extraction: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    config = {
        'colour_space': 'RGB',
        'discrim_bands': 3,
        'retention': 50.0
    }
    
    try:
        range_result = extract_colour_range_from_swatch("path/to/your/swatch.jpg", config)
        print(f"Color range: {range_result}")
    except Exception as e:
        print(f"Error: {str(e)}")
