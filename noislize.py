import cv2
import numpy as np
import re

def add_gaussian_noise(image, mean=0, std=20):
    """
    給圖片加上高斯雜訊
    image: 輸入的BGR圖片 (numpy array)
    mean: 雜訊均值
    std: 雜訊標準差
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

if __name__ == "__main__":
    imagePath = input("Enter the path to the image: ")
    img = cv2.imread(imagePath)  # 讀取圖片
    noisy_img = add_gaussian_noise(img, mean=0, std=30)
    noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)  # 轉換為gray格式
    cv2.imwrite(re.sub(r'_(.*?)\.(png|jpg)', r'_noisy.\2', imagePath), noisy_img)  # 儲存加雜訊的圖片