import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def create_synthetic_image(height=300, width=400):
    """Создать синтетическое изображение"""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        color = int(255 * (i / height))
        img[i, :, :] = (color, 255 - color, color // 2)

    cv2.circle(img, (width // 2, height // 2), 70, (0, 0, 255), -1)
    cv2.putText(img, "Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 3)
    return img


def add_noise(image, noise_type="gaussian", amount=0.02):
    """Добавить шум к изображению"""
    noisy = image.copy()
    if noise_type == "gaussian":
        mean, sigma = 0, 25
        gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy = cv2.add(image.astype(np.float32), gauss)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "s&p":
        noisy = image.copy()
        num_salt = np.ceil(amount * image.size * 0.5).astype(int)
        num_pepper = np.ceil(amount * image.size * 0.5).astype(int)

        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 255

        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 0
    return noisy


def estimate_noise(img):
    """Оценка уровня шума через дисперсию"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise_level = lap.var()
    return noise_level


def adaptive_denoise(img):
    """Адаптивная фильтрация на основе уровня шума"""
    noise_level = estimate_noise(img)

    if noise_level < 100:  
        # Низкий шум → легкая гауссовская фильтрация
        denoised = cv2.GaussianBlur(img, (3, 3), 0.5)
        method = "GaussianBlur (3x3)"
    elif noise_level < 500:
        # Средний шум → медианный фильтр
        denoised = cv2.medianBlur(img, 5)
        method = "MedianBlur (5)"
    else:
        # Сильный шум → bilateral для сохранения деталей
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        method = "Bilateral (9,75,75)"

    return denoised, method, noise_level


def compare_images(original, noisy, denoised, method, noise_level):
    """Визуальное сравнение"""
    images = [original, noisy, denoised]
    titles = ["Оригинал", f"С шумом (уровень≈{int(noise_level)})", f"Фильтр: {method}"]

    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    img = create_synthetic_image()
    noisy = add_noise(img, "gaussian")
    denoised, method, noise_level = adaptive_denoise(noisy)

    compare_images(img, noisy, denoised, method, noise_level)


if __name__ == "__main__":
    main()
