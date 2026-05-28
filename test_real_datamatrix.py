import cv2
import numpy as np
from src.datamatrix_scanner import AutoDataMatrixScanner
import time

# Создаём настоящий DataMatrix используя pyzbar/pylibdmtx или готовое изображение
def create_test_image_with_code():
    """Создаёт тестовое изображение с настоящим DataMatrix кодом"""
    # Генерируем простое чёрно-белое изображение с паттерном DataMatrix
    # Реальный DataMatrix имеет L-образный finder pattern и матрицу данных
    
    size = 300
    img = np.ones((size, size), dtype=np.uint8) * 255
    
    # L-образный finder pattern (левая и нижняя границы)
    img[:, :15] = 0  # Левая сплошная линия
    img[-15:, :] = 0  # Нижняя сплошная линия
    
    # Верхняя граница (пунктирная линия - timing pattern)
    for i in range(15, size-15, 20):
        img[:15, i:i+10] = 0
    
    # Правая граница (пунктирная линия - timing pattern)  
    for i in range(15, size-15, 20):
        img[i:i+10, -15:] = 0
    
    # Добавляем матрицу данных (симуляция)
    np.random.seed(42)
    for i in range(20, size-20, 15):
        for j in range(20, size-20, 15):
            if np.random.random() > 0.5:
                img[i:i+12, j:j+12] = 0
    
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def test_performance():
    """Тест производительности оптимизированного сканера"""
    print("=" * 60)
    print("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ OPTIMIZED SCANNER")
    print("=" * 60)
    
    scanner = AutoDataMatrixScanner(max_workers=4)
    test_img = create_test_image_with_code()
    
    # Тестируем несколько кадров для статистики
    times = []
    results = []
    
    print("\nОбработка 10 кадров...")
    for i in range(10):
        start = time.time()
        result = scanner.process_frame(test_img)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # мс
        results.append(result)
        
        print(f"Кадр {i+1}: {elapsed*1000:.2f} мс, статус: {result.status.value}, " +
              f"кэш хитов: {scanner.stats['cache_hits']}")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nСтатистика:")
    print(f"  Среднее время: {avg_time:.2f} мс")
    print(f"  Минимальное: {min_time:.2f} мс")
    print(f"  Максимальное: {max_time:.2f} мс")
    print(f"  Попаданий в кэш: {scanner.stats['cache_hits']}")
    print(f"  Параллельных декодирований: {scanner.stats['parallel_decodes']}")
    
    # Тест с разными изображениями (без кэша)
    print("\n" + "=" * 60)
    print("ТЕСТ БЕЗ КЭШИРОВАНИЯ (разные изображения)")
    print("=" * 60)
    
    scanner2 = AutoDataMatrixScanner(max_workers=4)
    times2 = []
    
    print("\nОбработка 5 разных изображений...")
    for i in range(5):
        # Создаём немного разные изображения
        img = create_test_image_with_code()
        # Добавляем небольшой шум для уникальности
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        start = time.time()
        result = scanner2.process_frame(img)
        elapsed = time.time() - start
        times2.append(elapsed * 1000)
        
        print(f"Изображение {i+1}: {elapsed*1000:.2f} мс, статус: {result.status.value}")
    
    avg_time2 = sum(times2) / len(times2)
    print(f"\nСреднее время без кэша: {avg_time2:.2f} мс")
    print(f"Попаданий в кэш: {scanner2.stats['cache_hits']}")
    
    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
    print("=" * 60)

if __name__ == "__main__":
    test_performance()
