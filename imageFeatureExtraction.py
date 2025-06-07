import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2

def transferToGray(imagePath):
    """
    Convert an image to grayscale.
    
    Parameters:
    image (numpy.ndarray): The input image in BGR format.
    
    Returns:
    numpy.ndarray: The grayscale image.
    """
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def waveletTransform(image, wavelet="db4", level=4):
    """
    Perform wavelet decomposition on an image and extract features.
    
    Parameters:
    image (numpy.ndarray): The input image in grayscale.
    wavelet (str): The type of wavelet to use for decomposition.
    level (int): The level of decomposition.
    
    Returns:
    list: Coefficients from the wavelet decomposition.
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def FeatureExtraction(coeffs, threshold=0.1):
    """
    Extract features from wavelet coefficients by thresholding.
    
    Parameters:
    coeffs (list): Coefficients from the wavelet decomposition.
    threshold (float): Threshold value for feature extraction.
    
    Returns:
    list: Extracted features.
    """
    features = [pywt.threshold(coeffs[0], threshold * np.max(coeffs[0]), mode='soft')]
    for detail in coeffs[1:]:
        feature_detail = tuple(pywt.threshold(c, threshold * np.max(c), mode='soft') for c in detail)
        features.append(feature_detail)
    return features

def reconstructImage(coeffs, wavelet="db4"):
    """
    Reconstruct an image from wavelet coefficients.
    
    Parameters:
    coeffs (list): Coefficients from the wavelet decomposition.
    wavelet (str): The type of wavelet used for decomposition.
    
    Returns:
    numpy.ndarray: The reconstructed image.
    """
    return pywt.waverec2(coeffs, wavelet)

def main():
    imagePath = input("Enter the path to the image: ")
    try:
        gray_image = transferToGray(imagePath)
        coeffs = waveletTransform(gray_image)
        features = FeatureExtraction(coeffs)

        # Display the original and reconstructed images
        reconstructed_image = reconstructImage(features)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Grayscale Image")
        plt.imshow(gray_image, cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Image")
        plt.imshow(reconstructed_image, cmap='gray')
        
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__": 
    main()