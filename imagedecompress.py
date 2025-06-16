def imageCoefficients(image):
    """
    Extract wavelet coefficients from a color image (RGB).
    Returns: list of coeffs for each channel [R, G, B]
    """
    import pywt
    import numpy as np

    image = image.convert('RGB')
    image_array = np.array(image)
    coeffs_list = []
    for ch in range(3):  # R, G, B
        coeffs = pywt.wavedec2(image_array[..., ch], 'db4', level=4)
        coeffs_list.append(coeffs)
    return coeffs_list

def imageSetHighFrequencyCoefficientsZeroCompression(coeffs_list):
    """
    Set the high frequency coefficients to be 0 for each channel.
    """
    import numpy as np
    compressed_coeffs_list = []
    for coeffs in coeffs_list:
        compressed_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:
                compressed_coeffs.append(coeff)
            else:
                compressed_coeffs.append(tuple(np.zeros_like(c) for c in coeff))
        compressed_coeffs_list.append(compressed_coeffs)
    return compressed_coeffs_list

def imageCoefficientsCompression(coeffs_list, threshold=0.1):
    """
    Using soft thresholding to compress the wavelet coefficients for each channel.
    """
    import numpy as np
    compressed_coeffs_list = []
    for coeffs in coeffs_list:
        compressed_coeffs = []
        for coeff in coeffs:
            if isinstance(coeff, tuple):
                compressed_coeffs.append(tuple(np.where(np.abs(c) < threshold, 0, c) for c in coeff))
            else:
                compressed_coeffs.append(np.where(np.abs(coeff) < threshold, 0, coeff))
        compressed_coeffs_list.append(compressed_coeffs)
    return compressed_coeffs_list

def reconstructImage(coeffs_list):
    """
    Reconstruct a color image from wavelet coefficients for each channel.
    """
    import pywt
    import numpy as np
    from PIL import Image
    channels = []
    for coeffs in coeffs_list:
        arr = pywt.waverec2(coeffs, 'db4')
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        channels.append(arr)
    # 對齊 shape（有時候重建後 shape 會略有不同）
    min_shape = np.min([c.shape for c in channels], axis=0)
    channels = [c[:min_shape[0], :min_shape[1]] for c in channels]
    img_array = np.stack(channels, axis=-1)
    return Image.fromarray(img_array, mode='RGB')

def main():
    imagePath = input("Enter the path to the image: ")
    from PIL import Image
    import re
    try:
        # Load the image
        image = Image.open(imagePath)

        # Extract wavelet coefficients for each channel
        coeffs_list = imageCoefficients(image)
        
        # Compress the coefficients by zeroing out high frequencies
        zeroHighFrequencycoefficent = imageSetHighFrequencyCoefficientsZeroCompression(coeffs_list)

        # Alternatively, you can use soft thresholding for compression
        normalized_coefficients = imageCoefficientsCompression(coeffs_list)

        # Reconstruct the image from compressed coefficients
        reconstructed_image = reconstructImage(zeroHighFrequencycoefficent)

        # Alternatively, you can reconstruct from normalized coefficients
        normalized_reconstructed_image = reconstructImage(normalized_coefficients)

        # Save the reconstructed image
        reconstructed_image.save(re.sub(r'_(.*?)\.(png|jpg)', r'_zero.\2', imagePath))
        normalized_reconstructed_image.save(re.sub(r'_(.*?)\.(png|jpg)', r'_normal.\2', imagePath))

        #sample_image = reconstructImage(coeffs_list)
        #sample_image.save(re.sub(r'_(.*?)\.(png|jpg)', r'_sample.\2', imagePath))

        for i in range(-5, 5):
            if( i % 2 == 1):
                special_coefficients = 5
            else:
                special_coefficients = 10
            
            special_coefficients = special_coefficients * (10 ** int(i/2))

            special_coefficients_image = imageCoefficientsCompression(coeffs_list, threshold=special_coefficients)
            special_reconstructed_image = reconstructImage(special_coefficients_image)
            newfilepath = re.sub(
                r'_(.*?)\.(png|jpg)',
                f'_{special_coefficients:.4f}.\\2',
                imagePath
            )

            special_reconstructed_image.save(newfilepath)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
