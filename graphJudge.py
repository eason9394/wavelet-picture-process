'''
the file is used to judge the quality of the image
'''

def ImageInput(imagePath):
    """
    Load an image from the specified path.
    
    Parameters:
    imagePath (str): The path to the image file.
    
    Returns:
    numpy.ndarray: The loaded image in BGR format.
    """
    import cv2
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    return image

'''
consider using the real world picture,
so the judgement don't have ground truth, reference assessment
'''

def ImagefeatureJudge(image):
    """
    Judge the quality of the image based on its features.
    
    Parameters:
    image (numpy.ndarray): The input image in BGR format.
    
    Returns:
    str: A string indicating the quality of the image.
    """
    import cv2
    import numpy as np
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the variance of the Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    '''
    returns a result of number
    '''

    return np.var(laplacian)

def ImageNoiseJudge(image):
    """
    Judge the noise level of the image based on the standard deviation of the Laplacian.
    
    Parameters:
    image (numpy.ndarray): The input image in BGR format.
    
    Returns:
    str: A string indicating the noise level of the image.
    """
    import cv2
    import numpy as np
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calculate the standard deviation
    return np.std(laplacian)

def main():
    imagePath = input("Enter the path to the image: ")
    try:
        image = ImageInput(imagePath)
        
        quality_score = ImagefeatureJudge(image)
        print(f"Image Feature Quality Score(higher score represent better): {quality_score}\n")
        
        noise_result = ImageNoiseJudge(image)
        print(f"Image Noise Level(higher score represent better): {noise_result}\n")


    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()