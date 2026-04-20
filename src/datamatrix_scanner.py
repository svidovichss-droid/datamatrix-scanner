"""
Модуль автоматического поиска, захвата и распознавания DataMatrix кодов

Авторы: А. Свидович / А. Петляков для PROGRESS
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import time
import threading


class DetectionStatus(Enum):
    """Статус детекции DataMatrix"""
    NOT_FOUND = "Не найден"
    FOUND = "Найден"
    DECODED = "Распознан"
    ERROR = "Ошибка"


@dataclass
class DataMatrixResult:
    """Результат обнаружения и распознавания DataMatrix"""
    status: DetectionStatus = DetectionStatus.NOT_FOUND
    data: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    confidence: float = 0.0
    roi: Optional[np.ndarray] = None
    timestamp: float = 0.0
    processing_time_ms: float = 0.0
    error_message: str = ""


class AutoDataMatrixScanner:
    """
    Автоматический сканер DataMatrix кодов
    
    Реализует:
    - Поиск области интереса (ROI) с DataMatrix
    - Автоматический захват лучшего кадра
    - Распознавание и декодирование
    - Верификацию результата
    """
    
    def __init__(self):
        # Параметры детекции
        self.min_code_size = 50  # Минимальный размер кода в пикселях
        self.max_code_size = 800  # Максимальный размер
        self.min_contrast = 30  # Минимальный контраст для детекции
        self.confidence_threshold = 0.5  # Порог уверенности детекции
        
        # Параметры захвата
        self.capture_timeout = 5.0  # Таймаут захвата (секунды)
        self.max_frames_buffer = 30  # Размер буфера кадров
        self.frame_stability_count = 3  # Количество стабильных кадров для захвата
        
        # Буфер кадров для анализа
        self._frame_buffer: List[np.ndarray] = []
        self._last_result: Optional[DataMatrixResult] = None
        self._lock = threading.Lock()
        
        # Статистика
        self.stats = {
            'frames_processed': 0,
            'codes_found': 0,
            'codes_decoded': 0,
            'avg_processing_time': 0.0
        }
    
    def process_frame(self, frame: np.ndarray) -> DataMatrixResult:
        """
        Обработка одиночного кадра
        
        Args:
            frame: Входное изображение (BGR)
            
        Returns:
            DataMatrixResult с результатами
        """
        start_time = time.time()
        result = DataMatrixResult(timestamp=start_time)
        
        try:
            if frame is None or frame.size == 0:
                result.status = DetectionStatus.ERROR
                result.error_message = "Пустой кадр"
                return result
            
            # Предобработка
            processed = self._preprocess_frame(frame)
            
            # Поиск DataMatrix
            detection = self._detect_datamatrix(processed)
            
            if not detection:
                result.status = DetectionStatus.NOT_FOUND
                self._update_stats(start_time, found=False)
                return result
            
            bbox, confidence, roi = detection
            
            # Проверка размера
            if not self._validate_size(bbox):
                result.status = DetectionStatus.NOT_FOUND
                self._update_stats(start_time, found=False)
                return result
            
            result.bbox = bbox
            result.confidence = confidence
            result.roi = roi.copy() if roi is not None else None
            
            # Попытка декодирования
            decoded_data = self._decode_roi(roi)
            
            if decoded_data:
                result.status = DetectionStatus.DECODED
                result.data = decoded_data
                self._update_stats(start_time, found=True, decoded=True)
            else:
                result.status = DetectionStatus.FOUND
                self._update_stats(start_time, found=True, decoded=False)
                
        except Exception as e:
            result.status = DetectionStatus.ERROR
            result.error_message = str(e)
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        self._last_result = result
        
        with self._lock:
            self.stats['frames_processed'] += 1
            prev_avg = self.stats['avg_processing_time']
            n = self.stats['frames_processed']
            self.stats['avg_processing_time'] = prev_avg + (result.processing_time_ms - prev_avg) / n
        
        return result
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Предобработка кадра для улучшения детекции
        
        Применяет:
        - Конвертацию в градации серого
        - Усиление контраста (CLAHE)
        - Уменьшение шума
        - Повышение резкости
        """
        # Конвертация в grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # CLAHE для усиления локального контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Уменьшение шума (быстрый фильтр)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Повышение резкости
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def _detect_datamatrix(self, image: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
        """
        Обнаружение DataMatrix на изображении
        
        Использует комбинированный подход:
        1. Детекция по L-образному паттерну (finder pattern)
        2. Детекция по квадратным контурам
        3. Детекция по текстуре
        
        Returns:
            Кортеж (bbox, confidence, roi) или None
        """
        candidates = []
        
        # Метод 1: Поиск по finder pattern (L-образный маркер)
        fp_candidates = self._detect_by_finder_pattern(image)
        if fp_candidates:
            candidates.extend(fp_candidates)
        
        # Метод 2: Поиск квадратных контуров
        contour_candidates = self._detect_by_contours(image)
        if contour_candidates:
            candidates.extend(contour_candidates)
        
        # Метод 3: Поиск по текстуре (для сложных случаев)
        texture_candidates = self._detect_by_texture(image)
        if texture_candidates:
            candidates.extend(texture_candidates)
        
        if not candidates:
            return None
        
        # Выбираем лучший кандидат
        best = max(candidates, key=lambda x: x[1])  # Сортируем по confidence
        bbox, confidence = best[:2]
        
        # Извлекаем ROI
        x, y, w, h = bbox
        padding = max(5, int(min(w, h) * 0.1))  # 10% отступ
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        roi = image[y1:y2, x1:x2]
        
        return bbox, confidence, roi
    
    def _detect_by_finder_pattern(self, image: np.ndarray) -> List[Tuple]:
        """
        Детекция по L-образному finder pattern DataMatrix
        
        DataMatrix имеет характерную L-образную границу слева и снизу
        """
        candidates = []
        
        # Бинаризация
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Морфологические операции для усиления линий
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Поиск контуров
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Фильтр по размеру
            if area < self.min_code_size * self.min_code_size * 0.5:
                continue
            if area > self.max_code_size * self.max_code_size * 2:
                continue
            
            # Аппроксимация полигоном
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Ищем прямоугольные формы
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Проверка на квадратность
                aspect_ratio = w / h if h > 0 else 0
                if 0.7 <= aspect_ratio <= 1.3:
                    # Оценка контраста внутри региона
                    roi = image[y:y+h, x:x+w]
                    contrast = np.std(roi)
                    
                    if contrast > self.min_contrast:
                        confidence = min(1.0, contrast / 100.0)
                        candidates.append(((x, y, w, h), confidence))
        
        return candidates
    
    def _detect_by_contours(self, image: np.ndarray) -> List[Tuple]:
        """
        Детекция по квадратным контурам с проверкой углов
        """
        candidates = []
        
        # Детекция краёв Canny
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Морфология для соединения разрывов
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Поиск контуров
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Фильтр по площади
            min_area = self.min_code_size * self.min_code_size * 0.3
            max_area = self.max_code_size * self.max_code_size
            if area < min_area or area > max_area:
                continue
            
            # Минимальный описанный прямоугольник
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            # Проверка на квадратность
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if 0.8 <= aspect_ratio <= 1.2:
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    w = int(width)
                    h = int(height)
                    
                    # Проверка границ изображения
                    if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
                        roi = image[y:y+h, x:x+w]
                        contrast = np.std(roi)
                        
                        if contrast > self.min_contrast * 0.8:
                            confidence = min(1.0, (contrast / 80.0) * (area / max_area) ** 0.3)
                            candidates.append(((x, y, w, h), confidence * 0.9))  # Немного снижаем вес
        
        return candidates
    
    def _detect_by_texture(self, image: np.ndarray) -> List[Tuple]:
        """
        Детекция по текстурным признакам (для сложных случаев)
        
        Использует анализ частотных характеристик
        """
        candidates = []
        
        # Скользящее окно для поиска областей с высокой частотой
        window_size = 64
        step = 32
        
        h, w = image.shape
        
        for y in range(0, h - window_size, step):
            for x in range(0, w - window_size, step):
                roi = image[y:y+window_size, x:x+window_size]
                
                # Вычисление дисперсии (мера текстуры)
                variance = np.var(roi)
                
                # DataMatrix имеет высокую частоту переходов
                if variance > 1000:  # Порог для текстурированной области
                    # Проверяем соседние окна для объединения
                    confidence = min(1.0, variance / 3000.0)
                    
                    # Создаём bounding box
                    bbox = (x, y, window_size, window_size)
                    candidates.append((bbox, confidence * 0.7))  # Низкий вес для этого метода
        
        # Объединение перекрывающихся регионов
        if candidates:
            candidates = self._merge_overlapping_boxes(candidates)
        
        return candidates
    
    def _merge_overlapping_boxes(self, boxes: List[Tuple]) -> List[Tuple]:
        """Объединение перекрывающихся bounding box"""
        if not boxes:
            return []
        
        # Сортировка по confidence
        boxes = sorted(boxes, key=lambda x: -x[1])
        
        merged = []
        used = [False] * len(boxes)
        
        for i, (bbox1, conf1) in enumerate(boxes):
            if used[i]:
                continue
            
            x1, y1, w1, h1 = bbox1
            total_conf = conf1
            count = 1
            
            # Ищем перекрывающиеся
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                
                x2, y2, w2, h2 = boxes[j][0]
                
                # Проверка перекрытия (IoU)
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w1, x2 + w2)
                inter_y2 = min(y1 + h1, y2 + h2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    # Перекрываются - объединяем
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    
                    x1, y1, w1, h1 = new_x, new_y, new_w, new_h
                    total_conf += boxes[j][1]
                    count += 1
                    used[j] = True
            
            avg_conf = total_conf / count
            merged.append(((x1, y1, w1, h1), avg_conf))
        
        return merged
    
    def _validate_size(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Проверка размера detected кода"""
        x, y, w, h = bbox
        size = min(w, h)
        return self.min_code_size <= size <= self.max_code_size
    
    def _decode_roi(self, roi: np.ndarray) -> Optional[str]:
        """
        Декодирование DataMatrix из ROI
        
        Использует несколько методов для повышения надёжности:
        1. Прямое декодирование pyzbar
        2. Декодирование с различными уровнями бинаризации
        3. Декодирование с коррекцией перспективы
        """
        if roi is None or roi.size == 0:
            return None
        
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            
            # Метод 1: Прямое декодирование
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            decoded = pyzbar_decode(gray, symbols=[2])  # 2 = DataMatrix
            if decoded:
                return decoded[0].data.decode('utf-8', errors='ignore')
            
            # Метод 2: Попытка с адаптивной бинаризацией
            for block_size in [11, 21, 31]:
                binary = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    block_size,
                    2
                )
                decoded = pyzbar_decode(binary, symbols=[2])
                if decoded:
                    return decoded[0].data.decode('utf-8', errors='ignore')
            
            # Метод 3: Инверсия
            inverted = cv2.bitwise_not(gray)
            decoded = pyzbar_decode(inverted, symbols=[2])
            if decoded:
                return decoded[0].data.decode('utf-8', errors='ignore')
            
        except ImportError:
            pass
        except Exception as e:
            print(f"Ошибка декодирования: {e}")
        
        return None
    
    def _update_stats(self, start_time: float, found: bool = False, decoded: bool = False):
        """Обновление статистики"""
        if found:
            self.stats['codes_found'] += 1
        if decoded:
            self.stats['codes_decoded'] += 1
    
    def get_last_result(self) -> Optional[DataMatrixResult]:
        """Получение последнего результата"""
        with self._lock:
            return self._last_result.copy() if self._last_result else None
    
    def get_stats(self) -> Dict:
        """Получение статистики работы"""
        with self._lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """Сброс статистики"""
        with self._lock:
            self.stats = {
                'frames_processed': 0,
                'codes_found': 0,
                'codes_decoded': 0,
                'avg_processing_time': 0.0
            }


class ContinuousScanner:
    """
    Непрерывный сканер DataMatrix с автоматическим захватом
    
    Работает в отдельном потоке, постоянно анализируя видеопоток
    и автоматически захватывая лучший кадр при обнаружении кода
    """
    
    def __init__(self, scanner: AutoDataMatrixScanner):
        self.scanner = scanner
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback = None
        self._auto_capture_enabled = False
        self._capture_callback = None
        
        # Параметры автозахвата
        self.stable_frames_required = 5  # Количество стабильных кадров
        self.quality_threshold = 0.7  # Минимальное качество для захвата
        
        # Состояние
        self._consecutive_detections = 0
        self._best_frame = None
        self._best_result = None
    
    def start(self, callback=None):
        """
        Запуск непрерывного сканирования
        
        Args:
            callback: Функция обратного вызова при успешном распознавании
        """
        if self._running:
            return
        
        self._running = True
        self._callback = callback
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Остановка сканирования"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._consecutive_detections = 0
        self._best_frame = None
    
    def set_auto_capture(self, enabled: bool, capture_callback=None):
        """
        Включение/выключение автоматического захвата
        
        Args:
            enabled: Включить автозахват
            capture_callback: Callback для сохранения кадра
        """
        self._auto_capture_enabled = enabled
        self._capture_callback = capture_callback
    
    def _scan_loop(self):
        """Основной цикл сканирования"""
        while self._running:
            # Здесь должен быть вызов process_frame с текущим кадром
            # Кадр должен поступать из внешнего источника (камеры)
            time.sleep(0.01)  # Ожидание кадра
    
    def process_frame_async(self, frame: np.ndarray) -> Optional[DataMatrixResult]:
        """
        Асинхронная обработка кадра
        
        Должен вызываться из потока камеры
        """
        if not self._running:
            return None
        
        result = self.scanner.process_frame(frame)
        
        if result.status == DetectionStatus.DECODED:
            self._consecutive_detections += 1
            
            # Сохраняем лучший результат
            if self._best_result is None or result.confidence > self._best_result.confidence:
                self._best_frame = frame.copy()
                self._best_result = result
            
            # Проверка условия автозахвата
            if (self._auto_capture_enabled and 
                self._consecutive_detections >= self.stable_frames_required and
                result.confidence >= self.quality_threshold):
                
                if self._capture_callback:
                    self._capture_callback(self._best_frame, self._best_result)
                
                # Сброс после захвата
                self._consecutive_detections = 0
                self._best_frame = None
            
            # Callback при успешном распознавании
            if self._callback:
                try:
                    self._callback(result)
                except Exception as e:
                    print(f"Callback error: {e}")
        else:
            self._consecutive_detections = 0
        
        return result


def create_scanner() -> AutoDataMatrixScanner:
    """Фабричная функция для создания сканера"""
    return AutoDataMatrixScanner()


def create_continuous_scanner() -> ContinuousScanner:
    """Фабричная функция для создания непрерывного сканера"""
    return ContinuousScanner(AutoDataMatrixScanner())
