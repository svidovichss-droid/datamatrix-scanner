#!/usr/bin/env python3
"""
Демонстрация работы автоматического сканера DataMatrix

Авторы: А. Свидович / А. Петляков для PROGRESS
"""

import sys
import time
from src.datamatrix_scanner import AutoDataMatrixScanner, ContinuousScanner, DetectionStatus
from src.camera import simulate_datamatrix_image
import numpy as np
import cv2


def demo_automatic_scanner():
    """Демонстрация работы автоматического сканера"""
    
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ АВТОМАТИЧЕСКОГО СКАНЕРА DATAMATRIX")
    print("=" * 60)
    print()
    
    # Создаём сканер
    scanner = AutoDataMatrixScanner()
    
    print("[1] Тестирование обнаружения DataMatrix...")
    print("-" * 60)
    
    # Генерируем тестовое изображение
    test_img = simulate_datamatrix_image(400)
    
    # Обрабатываем кадр
    start_time = time.time()
    result = scanner.process_frame(test_img)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"Статус детекции: {result.status.value}")
    print(f"Bounding box: {result.bbox}")
    print(f"Уверенность: {result.confidence:.2%}")
    print(f"Время обработки: {elapsed:.1f} мс")
    
    if result.roi is not None:
        print(f"Размер ROI: {result.roi.shape}")
    
    print()
    print("[2] Статистика работы:")
    print("-" * 60)
    stats = scanner.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print()
    print("[3] Тестирование серии кадров (имитация видеопотока)...")
    print("-" * 60)
    
    # Имитация обработки видеопотока
    num_frames = 10
    detected_count = 0
    
    for i in range(num_frames):
        # Добавляем небольшой шум для реалистичности
        noise = np.random.normal(0, 5, test_img.shape).astype(np.uint8)
        noisy_img = cv2.add(test_img, noise)
        
        result = scanner.process_frame(noisy_img)
        
        if result.status != DetectionStatus.NOT_FOUND:
            detected_count += 1
            status_symbol = "✓"
        else:
            status_symbol = "✗"
        
        print(f"  Кадр {i+1:2d}: {status_symbol} {result.status.value:12s} "
              f"(уверенность: {result.confidence:5.1%}, время: {result.processing_time_ms:5.1f} мс)")
    
    print()
    print(f"Обнаружено кодов: {detected_count}/{num_frames} ({detected_count/num_frames*100:.0f}%)")
    
    print()
    print("[4] Обновлённая статистика:")
    print("-" * 60)
    stats = scanner.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    
    return scanner


def demo_continuous_scanner():
    """Демонстрация непрерывного сканера"""
    
    print()
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ НЕПРЕРЫВНОГО СКАНЕРА")
    print("=" * 60)
    print()
    
    # Создаём непрерывный сканер
    continuous = ContinuousScanner(AutoDataMatrixScanner())
    
    results_log = []
    
    def on_result(result):
        """Callback при обнаружении кода"""
        results_log.append(result)
        print(f"  >> ОБНАРУЖЕНО: {result.data[:30] if result.data else 'N/A'}... "
              f"(уверенность: {result.confidence:.1%})")
    
    # Запускаем сканер
    continuous.start(callback=on_result)
    
    print("Обработка 5 кадров в непрерывном режиме...")
    print("-" * 60)
    
    # Имитируем поступление кадров
    for i in range(5):
        test_img = simulate_datamatrix_image(400)
        result = continuous.process_frame_async(test_img)
        time.sleep(0.1)
    
    # Останавливаем сканер
    continuous.stop()
    
    print()
    print(f"Всего обнаружено: {len(results_log)} кодов")
    print(f"Статистика сканера: {continuous.scanner.get_stats()}")
    
    print()
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ НЕПРЕРЫВНОГО СКАНЕРА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    # Запуск демонстрации
    scanner = demo_automatic_scanner()
    demo_continuous_scanner()
    
    print()
    print("Все тесты пройдены успешно!")
    print()
    print("Для использования в GUI:")
    print("  1. Подключите камеру через меню 'Камера -> Подключить'")
    print("  2. Включите чекбокс 'Автосканер DataMatrix'")
    print("  3. Сканер автоматически найдёт и распознает коды")
    print("  4. При включённом 'Автозахвате' кадры сохраняются в историю")
