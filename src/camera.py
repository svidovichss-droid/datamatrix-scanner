"""
Модуль работы с промышленной камерой

Авторы: А. Свидович / А. Петляков для PROGRESS
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time


class CameraStatus(Enum):
    """Статус камеры"""
    DISCONNECTED = "Отключена"
    CONNECTING = "Подключение..."
    CONNECTED = "Подключена"
    STREAMING = "Трансляция"
    ERROR = "Ошибка"


@dataclass
class CameraConfig:
    """Конфигурация камеры"""
    device_id: int = 0
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    brightness: int = 128
    contrast: int = 128
    gain: int = 100
    exposure: int = 100
    auto_exposure: bool = False
    white_balance: str = "auto"


class CameraManager:
    """
    Менеджер промышленной камеры

    Поддерживает:
    - USB камеры
    - IP камеры
    - Камеры машинного зрения (через промышленные SDK)
    """

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.config = CameraConfig()
        self.status = CameraStatus.DISCONNECTED
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._callbacks = []
        self._last_error = ""

    @property
    def is_connected(self) -> bool:
        """Проверка подключения"""
        return self.cap is not None and self.cap.isOpened()

    @property
    def is_streaming(self) -> bool:
        """Проверка активной трансляции"""
        return self._streaming

    @property
    def current_frame(self) -> Optional[np.ndarray]:
        """Получение текущего кадра"""
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def connect(self, device_id: int = None) -> bool:
        """
        Подключение к камере

        Args:
            device_id: ID устройства (для USB камер) или URL (для IP камер)

        Returns:
            True при успешном подключении
        """
        self.status = CameraStatus.CONNECTING

        try:
            # Закрываем предыдущее подключение
            self.disconnect()

            if device_id is not None:
                self.config.device_id = device_id

            # Проверяем, может ли это быть IP камерой
            if isinstance(device_id, str) and (
                device_id.startswith('http://') or
                device_id.startswith('https://') or
                device_id.startswith('rtsp://')
            ):
                url = device_id
            else:
                # USB камера
                url = self.config.device_id if device_id is None else device_id

            # Открытие камеры
            self.cap = cv2.VideoCapture(url)

            if not self.cap.isOpened():
                self.status = CameraStatus.ERROR
                self._last_error = "Не удалось открыть камеру"
                return False

            # Применение конфигурации
            self._apply_config()

            self.status = CameraStatus.CONNECTED
            return True

        except Exception as e:
            self.status = CameraStatus.ERROR
            self._last_error = str(e)
            return False

    def disconnect(self):
        """Отключение от камеры"""
        self.stop_streaming()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.status = CameraStatus.DISCONNECTED

    def _apply_config(self):
        """Применение конфигурации к камере"""
        if self.cap is None:
            return

        cap = self.cap

        # Разрешение
        w, h = self.config.resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # FPS
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        # Яркость (0-255)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)

        # Контраст (0-255)
        cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)

        # Усиление (gain)
        if hasattr(cv2, 'CAP_PROP_GAIN'):
            cap.set(cv2.CAP_PROP_GAIN, self.config.gain)

        # Экспозиция
        if hasattr(cv2, 'CAP_PROP_EXPOSURE'):
            if self.config.auto_exposure:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 = 3/4 = auto
            else:
                cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure / 1000.0)

    def update_config(self, **kwargs):
        """Обновление конфигурации"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        if self.is_connected:
            self._apply_config()

    def start_streaming(self, callback: Callable[[np.ndarray], None] = None) -> bool:
        """
        Запуск потоковой трансляции

        Args:
            callback: Функция обратного вызова для каждого кадра

        Returns:
            True при успешном запуске
        """
        if not self.is_connected:
            self._last_error = "Камера не подключена"
            return False

        if self._streaming:
            return True

        self._streaming = True

        if callback:
            self._callbacks.append(callback)

        # Запуск потока чтения
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

        self.status = CameraStatus.STREAMING
        return True

    def stop_streaming(self):
        """Остановка трансляции"""
        self._streaming = False

        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None

        self._callbacks.clear()

        if self.is_connected:
            self.status = CameraStatus.CONNECTED

    def _stream_loop(self):
        """Основной цикл трансляции"""
        while self._streaming and self.cap is not None:
            try:
                ret, frame = self.cap.read()

                if ret:
                    # Сохраняем текущий кадр
                    with self._frame_lock:
                        self._current_frame = frame

                    # Вызываем колбэки
                    for callback in self._callbacks:
                        try:
                            callback(frame)
                        except Exception as e:
                            print(f"Callback error: {e}")

                else:
                    time.sleep(0.01)

            except Exception as e:
                print(f"Stream error: {e}")
                self._last_error = str(e)
                break

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Захват одиночного кадра

        Returns:
            Изображение или None
        """
        if not self.is_connected:
            return None

        ret, frame = self.cap.read()
        return frame if ret else None

    def set_roi(self, x: int, y: int, w: int, h: int):
        """Установка Region of Interest (область интереса)"""
        if self.cap is None:
            return

        # OpenCV не поддерживает ROI для всех камер
        # Это можно использовать при обработке изображения
        pass

    def get_available_cameras(self) -> list:
        """
        Поиск доступных камер

        Returns:
            Список ID доступных камер
        """
        available = []

        # Проверяем USB камеры
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
            except:
                pass

        return available

    def get_last_error(self) -> str:
        """Получение последней ошибки"""
        return self._last_error


class ImageAcquisition:
    """
    Класс для захвата и предварительной обработки изображений DataMatrix
    """

    def __init__(self, camera: CameraManager):
        self.camera = camera

    def capture_and_preprocess(self) -> Optional[np.ndarray]:
        """
        Захват и предобработка изображения

        Returns:
            Предобработанное изображение или None
        """
        frame = self.camera.capture_frame()
        if frame is None:
            return None

        return self.preprocess_image(frame)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для улучшения распознавания

        Args:
            image: Входное изображение

        Returns:
            Обработанное изображение
        """
        # Конвертация в градации серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Усиление контраста (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Уменьшение шума
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Повышение резкости
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        return sharpened

    def auto_adjust_exposure(self, target_brightness: int = 128, iterations: int = 5):
        """
        Автоматическая настройка экспозиции

        Args:
            target_brightness: Целевая яркость (0-255)
            iterations: Максимальное количество итераций
        """
        for _ in range(iterations):
            frame = self.camera.capture_frame()
            if frame is None:
                break

            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            mean_brightness = np.mean(gray)

            # Корректировка экспозиции
            diff = target_brightness - mean_brightness

            if abs(diff) < 5:
                break

            current_exposure = self.camera.config.exposure
            new_exposure = int(current_exposure * (1 + diff / 255))
            new_exposure = max(1, min(1000, new_exposure))

            self.camera.update_config(exposure=new_exposure)

            time.sleep(0.1)


def simulate_datamatrix_image(size: int = 300) -> np.ndarray:
    """
    Генерация тестового изображения DataMatrix кода

    Используется для отладки без реальной камеры
    """
    import random

    # Создаём пустое изображение
    img = np.ones((size, size), dtype=np.uint8) * 255

    # Рисуем рамку
    border = size // 20
    cv2.rectangle(img, (border, border), (size-border, size-border), 0, 2)

    # Генерируем случайный узор (имитация DataMatrix)
    cell_size = size // 15
    margin = border + cell_size

    # Рисуем "finder pattern" углы
    corner_size = cell_size * 3
    # Левый верхний
    cv2.rectangle(img, (margin, margin),
                  (margin + corner_size, margin + corner_size), 0, -1)
    cv2.rectangle(img, (margin + cell_size, margin + cell_size),
                  (margin + corner_size - cell_size, margin + corner_size - cell_size), 255, -1)
    cv2.rectangle(img, (margin + cell_size*2, margin + cell_size*2),
                  (margin + corner_size - cell_size*2, margin + corner_size - cell_size*2), 0, -1)

    # Правый верхний
    x = size - margin - corner_size
    cv2.rectangle(img, (x, margin),
                  (x + corner_size, margin + corner_size), 0, -1)
    cv2.rectangle(img, (x + cell_size, margin + cell_size),
                  (x + corner_size - cell_size, margin + corner_size - cell_size), 255, -1)
    cv2.rectangle(img, (x + cell_size*2, margin + cell_size*2),
                  (x + corner_size - cell_size*2, margin + corner_size - cell_size*2), 0, -1)

    # Левый нижний
    y = size - margin - corner_size
    cv2.rectangle(img, (margin, y),
                  (margin + corner_size, y + corner_size), 0, -1)
    cv2.rectangle(img, (margin + cell_size, y + cell_size),
                  (margin + corner_size - cell_size, y + corner_size - cell_size), 255, -1)
    cv2.rectangle(img, (margin + cell_size*2, y + cell_size*2),
                  (margin + corner_size - cell_size*2, y + corner_size - cell_size*2), 0, -1)

    # Заполняем остальное случайными модулями
    for i in range(5, size // cell_size - 1):
        for j in range(5, size // cell_size - 1):
            if random.random() > 0.5:
                x1 = i * cell_size
                y1 = j * cell_size
                cv2.rectangle(img, (x1, y1),
                            (x1 + cell_size - 1, y1 + cell_size - 1), 0, -1)

    # Добавляем шум
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Конвертируем в цветное для совместимости
    colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return colored
