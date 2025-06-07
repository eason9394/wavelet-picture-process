def edgeDetection(image):
    """
    Applies edge detection to the input image using the Canny method.
    
    Parameters:
    image (numpy.ndarray): The input image in BGR format.
    
    Returns:
    numpy.ndarray: The image with edges detected.
    """
    import cv2
    import numpy as np

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)
    
    return edges

def main():
    # Load an image (replace 'image.jpg' with your image path)
    import cv2
    image = cv2.imread(input("Enter the path to the image: "))
    if image is None:
        print("Error: Could not load image.")
        return

    # Apply edge detection
    edges = edgeDetection(image)

    # Display the original and edge-detected images
    cv2.imshow('Original Image', image)
    cv2.imshow('Edge Detected Image', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()