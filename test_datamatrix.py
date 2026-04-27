import cv2
import numpy as np
from src.datamatrix_scanner import AutoDataMatrixScanner

# Создаём тестовый DataMatrix код (синтетический паттерн)
def create_test_datamatrix():
    """Создаёт синтетическое изображение, похожее на DataMatrix"""
    size = 200
    img = np.ones((size, size), dtype=np.uint8) * 255
    
    # Рисуем L-образный finder pattern (как в DataMatrix)
    # Левая граница (сплошная линия)
    img[:, 10:20] = 0
    # Нижняя граница (сплошная линия)  
    img[size-20:size-10, :] = 0
    
    # Рисуем сетку модулей (псевдо-данные)
    for i in range(10, size-20, 10):
        for j in range(20, size-10, 10):
            if (i + j) % 20 == 0:
                img[i:i+8, j:j+8] = 0
    
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Тест 1: Синтетический DataMatrix
print("Тест 1: Синтетический DataMatrix")
scanner = AutoDataMatrixScanner()
test_img = create_test_datamatrix()
result = scanner.process_frame(test_img)
print(f"Статус: {result.status.value}")
print(f"Данные: {result.data if result.data else 'Нет данных'}")
print(f"Confidence: {result.confidence:.2f}")
print(f"BBox: {result.bbox}")
print(f"Время обработки: {result.processing_time_ms:.2f} мс")
print()

# Тест 2: Пустой кадр
print("Тест 2: Пустой кадр")
empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
result = scanner.process_frame(empty_img)
print(f"Статус: {result.status.value}")
print()

# Тест 3: Кадр с шумом
print("Тест 3: Кадр с шумом")
noise_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = scanner.process_frame(noise_img)
print(f"Статус: {result.status.value}")
print()

# Тест 4: Простое изображение с высоким контрастом
print("Тест 4: Простое изображение с высоким контрастом")
contrast_img = np.zeros((200, 200), dtype=np.uint8)
contrast_img[50:150, 50:150] = 255
contrast_img_bgr = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2BGR)
result = scanner.process_frame(contrast_img_bgr)
print(f"Статус: {result.status.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"BBox: {result.bbox}")
print()

print("Все тесты завершены!")
