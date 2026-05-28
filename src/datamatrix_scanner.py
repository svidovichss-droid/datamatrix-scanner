"""
Модуль автоматического поиска, захвата и распознавания DataMatrix кодов

Авторы: А. Свидович / А. Петляков для PROGRESS
Оптимизированная версия с ускоренной обработкой и декодированием
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import hashlib


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
    
    Оптимизации производительности:
    - Кэширование результатов обработки
    - Пул потоков для параллельного декодирования
    - Умный выбор стратегий обработки
    - Раннее завершение при успешном декодировании
    """
    
    # Классовые переменные для общих ресурсов
    _decoder_pool = None
    _clahe_cache = None
    
    def __init__(self, max_workers: int = 4):
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
        
        # Оптимизация: кэш последних обработанных кадров
        self._cache_max_size = 10
        self._frame_cache = deque(maxlen=self._cache_max_size)
        self._cache_lock = threading.Lock()
        
        # Оптимизация: пул потоков для параллельного декодирования
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Оптимизация: предсоздание CLAHE объекта
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Статистика
        self.stats = {
            'frames_processed': 0,
            'codes_found': 0,
            'codes_decoded': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'parallel_decodes': 0
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
            
            # Оптимизация: проверка кэша по хэшу кадра
            frame_hash = self._fast_frame_hash(frame)
            cached_result = self._get_from_cache(frame_hash)
            if cached_result is not None:
                with self._lock:
                    self.stats['cache_hits'] += 1
                return cached_result
            
            # Предобработка (оптимизированная)
            processed = self._preprocess_frame(frame)
            
            # Поиск DataMatrix (оптимизированный поиск)
            detection = self._detect_datamatrix(processed)
            
            if not detection:
                result.status = DetectionStatus.NOT_FOUND
                self._update_stats(start_time, found=False)
                self._add_to_cache(frame_hash, result)
                return result
            
            bbox, confidence, roi = detection
            
            # Проверка размера
            if not self._validate_size(bbox):
                result.status = DetectionStatus.NOT_FOUND
                self._update_stats(start_time, found=False)
                self._add_to_cache(frame_hash, result)
                return result
            
            result.bbox = bbox
            result.confidence = confidence
            result.roi = roi.copy() if roi is not None else None
            
            # Попытка декодирования (оптимизированная)
            decoded_data = self._decode_roi_optimized(roi)
            
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
        
        # Кэширование результата
        self._add_to_cache(frame_hash, result)
        
        with self._lock:
            self.stats['frames_processed'] += 1
            prev_avg = self.stats['avg_processing_time']
            n = self.stats['frames_processed']
            self.stats['avg_processing_time'] = prev_avg + (result.processing_time_ms - prev_avg) / n
        
        return result
    
    def _fast_frame_hash(self, frame: np.ndarray) -> int:
        """Быстрое вычисление хэша кадра для кэширования"""
        # Используем уменьшенную версию кадра для скорости
        if frame.shape[0] > 64 or frame.shape[1] > 64:
            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        
        # Быстрый хэш на основе суммы пикселей и формы
        return hash((small.shape, small.sum() % 1000000, small.mean() % 1000))
    
    def _get_from_cache(self, frame_hash: int) -> Optional[DataMatrixResult]:
        """Получение результата из кэша"""
        with self._cache_lock:
            for cached_hash, cached_result in self._frame_cache:
                if cached_hash == frame_hash:
                    return cached_result
        return None
    
    def _add_to_cache(self, frame_hash: int, result: DataMatrixResult):
        """Добавление результата в кэш"""
        with self._cache_lock:
            self._frame_cache.append((frame_hash, result))
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Предобработка кадра для улучшения детекции
        
        Применяет:
        - Конвертацию в градации серого
        - Быстрое усиление контраста
        - Минимальное уменьшение шума
        
        Оптимизации:
        - Использование предсозданного CLAHE объекта
        - SIMD-оптимизированные операции OpenCV
        """
        # Конвертация в grayscale (используем оптимизированный метод)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Быстрое усиление контраста через предсозданный CLAHE объект
        enhanced = self._clahe.apply(gray)
        
        # Очень быстрое уменьшение шума (медианный фильтр с малым ядром)
        denoised = cv2.medianBlur(enhanced, 3)
        
        return denoised
    
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
        Декодирование DataMatrix из ROI (устаревший метод для совместимости)
        
        Использует расширенный набор методов для максимального повышения надёжности:
        1. Прямое декодирование pyzbar
        2. Декодирование pylibdmtx (специализированная библиотека)
        3. Множественные стратегии бинаризации
        4. Коррекция перспективы и поворота
        5. Улучшение контраста и резкости
        6. Масштабирование для оптимального размера
        7. Комбинированные методы предобработки
        """
        return self._decode_roi_optimized(roi)
    
    def _decode_roi_optimized(self, roi: np.ndarray) -> Optional[str]:
        """
        Оптимизированное декодирование DataMatrix из ROI
        
        Ключевые оптимизации:
        1. Приоритет быстрых стратегий
        2. Параллельная обработка стратегий в пуле потоков
        3. Раннее завершение при успехе
        4. Умный выбор стратегий на основе характеристик ROI
        5. Кэширование результатов декодирования
        """
        if roi is None or roi.size == 0:
            return None
        
        decoded_data = None
        
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            
            # Подготовка grayscale изображения
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()
            
            # Быстрая попытка №1: прямое декодирование оригинала
            decoded = pyzbar_decode(gray, symbols=[2])
            if decoded:
                return decoded[0].data.decode('utf-8', errors='ignore')
            
            # Быстрая попытка №2: инверсия
            inverted = cv2.bitwise_not(gray)
            decoded = pyzbar_decode(inverted, symbols=[2])
            if decoded:
                return decoded[0].data.decode('utf-8', errors='ignore')
            
            # Быстрая попытка №3: CLAHE
            clahe_img = self._clahe.apply(gray)
            decoded = pyzbar_decode(clahe_img, symbols=[2])
            if decoded:
                return decoded[0].data.decode('utf-8', errors='ignore')
            
            # Анализ ROI для выбора оптимальных стратегий
            strategies = self._generate_smart_strategies(gray)
            
            # Параллельное декодирование стратегий
            decoded_data = self._parallel_decode(strategies)
            
            if decoded_data:
                return decoded_data
            
            # Финальная попытка с коррекцией перспективы (только если простые методы не сработали)
            perspective_corrected = self._correct_perspective(gray)
            if perspective_corrected is not None:
                decoded = pyzbar_decode(perspective_corrected, symbols=[2])
                if decoded:
                    return decoded[0].data.decode('utf-8', errors='ignore')
                
                # Параллельное декодирование для перспективно исправленного изображения
                perspective_strategies = self._generate_smart_strategies(perspective_corrected)
                decoded_data = self._parallel_decode(perspective_strategies)
                if decoded_data:
                    return decoded_data
                        
        except ImportError:
            pass
        except Exception as e:
            print(f"Ошибка декодирования: {e}")
        
        return None
    
    def _parallel_decode(self, strategies: List[Tuple[str, np.ndarray]]) -> Optional[str]:
        """
        Параллельное декодирование множества стратегий обработки
        
        Args:
            strategies: Список кортежей (название, изображение)
            
        Returns:
            Декодированные данные или None
        """
        if not strategies:
            return None
        
        # Для небольшого количества стратегий используем последовательное декодирование
        if len(strategies) <= 4:
            for name, img in strategies:
                result = self._try_decode_single(img)
                if result:
                    return result
            return None
        
        # Для большого количества - параллельное выполнение
        # Не используем with, чтобы не закрывать executor
        futures = {}
        try:
            for name, img in strategies[:8]:  # Ограничиваем количество для скорости
                future = self._executor.submit(self._try_decode_single, img)
                futures[future] = name
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=0.5)  # Таймаут на каждую стратегию
                    if result:
                        with self._lock:
                            self.stats['parallel_decodes'] += 1
                        return result
                except Exception:
                    continue
        except RuntimeError:
            # Executor закрыт, используем последовательное декодирование
            for name, img in strategies:
                result = self._try_decode_single(img)
                if result:
                    return result
        
        return None
    
    def _try_decode_single(self, image: np.ndarray) -> Optional[str]:
        """
        Попытка декодирования одного изображения через pyzbar и pylibdmtx
        
        Args:
            image: Изображение для декодирования
            
        Returns:
            Декодированные данные или None
        """
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            
            decoded = pyzbar_decode(image, symbols=[2])
            if decoded:
                return decoded[0].data.decode('utf-8', errors='ignore')
            
            # Пробуем pylibdmtx как запасной вариант
            try:
                from pylibdmtx.pylibdmtx import decode as dmtx_decode
                decoded = dmtx_decode(image)
                if decoded:
                    return decoded[0].data.decode('utf-8', errors='ignore')
            except:
                pass
                
        except:
            pass
        
        return None
    
    def _generate_smart_strategies(self, gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Умная генерация стратегий предобработки на основе характеристик изображения
        
        Возвращает список кортежей (название, изображение) приоритезированный по вероятности успеха
        """
        strategies = []
        
        # Вычисляем метрики изображения для выбора стратегий
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Стратегия 1: Адаптивная бинаризация (эффективна при неравномерном освещении)
        if std_val < 50:  # Низкий контраст
            for block_size in [21, 31, 51]:
                binary = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    5
                )
                strategies.append((f"adaptive_gauss_{block_size}", binary))
        
        # Стратегия 2: Оцу бинаризация (эффективна при хорошем контрасте)
        if std_val >= 50:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            strategies.append(("otsu", binary_otsu))
            strategies.append(("otsu_inv", cv2.bitwise_not(binary_otsu)))
        
        # Стратегия 3: Масштабирование для маленьких кодов
        if gray.shape[0] < 100 or gray.shape[1] < 100:
            scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            strategies.append(("scaled_2x", scaled))
            scaled_clahe = self._clahe.apply(scaled)
            strategies.append(("scaled_clahe", scaled_clahe))
        
        # Стратегия 4: Усиление резкости для размытых изображений
        if std_val > 30:
            sharpened = self._sharpen_image(gray, 1.5)
            strategies.append(("sharpened", sharpened))
            strategies.append(("sharpened_inv", cv2.bitwise_not(sharpened)))
        
        # Стратегия 5: CLAHE + адаптивная бинаризация (комбинированная)
        clahe_img = self._clahe.apply(gray)
        binary_combined = cv2.adaptiveThreshold(
            clahe_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5
        )
        strategies.append(("clahe_adaptive", binary_combined))
        
        # Стратегия 6: Инвертированные версии для тёмных кодов на светлом фоне
        if mean_val > 128:  # Светлое изображение
            strategies.append(("inverted", cv2.bitwise_not(gray)))
            strategies.append(("clahe_inverted", cv2.bitwise_not(clahe_img)))
        
        return strategies
    
    def _generate_processing_strategies(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Генерация множества стратегий предобработки изображения
        
        Возвращает список обработанных изображений для попытки декодирования
        """
        strategies = []
        
        # Стратегия 1: Оригинал
        strategies.append(gray.copy())
        
        # Стратегия 2: Инверсия
        strategies.append(cv2.bitwise_not(gray))
        
        # Стратегия 3: Усиление контраста (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        strategies.append(clahe.apply(gray))
        
        # Стратегия 4: CLAHE + инверсия
        clahe_img = clahe.apply(gray)
        strategies.append(cv2.bitwise_not(clahe_img))
        
        # Стратегия 5-8: Адаптивная бинаризация с разными параметрами
        for block_size in [11, 21, 31, 51]:
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                5
            )
            strategies.append(binary)
            
            # Инвертированная адаптивная бинаризация
            strategies.append(cv2.bitwise_not(binary))
        
        # Стратегия 9-12: Адаптивная бинаризация с THRESH_MEAN_C
        for block_size in [15, 25, 35, 45]:
            binary_mean = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block_size,
                7
            )
            strategies.append(binary_mean)
        
        # Стратегия 13-16: Бинаризация Оцу с различными предобработками
        for blur_kernel in [3, 5, 7, 9]:
            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            strategies.append(binary_otsu)
            
            # Инвертированная Оцу
            _, binary_otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            strategies.append(binary_otsu_inv)
        
        # Стратегия 17-20: Улучшение резкости
        for alpha in [1.3, 1.5, 1.7, 2.0]:
            enhanced = self._sharpen_image(gray, alpha)
            strategies.append(enhanced)
            strategies.append(cv2.bitwise_not(enhanced))
        
        # Стратегия 21-24: Масштабирование для разных размеров
        for scale in [1.5, 2.0, 2.5, 3.0]:
            if gray.shape[0] < 200 or gray.shape[1] < 200:
                scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                strategies.append(scaled)
                
                # CLAHE на масштабированном
                clahe_scaled = clahe.apply(scaled)
                strategies.append(clahe_scaled)
        
        # Стратегия 25-28: Комбинация CLAHE + sharpening
        for alpha in [1.3, 1.5, 1.7, 2.0]:
            clahe_img = clahe.apply(gray)
            sharpened = self._sharpen_image(clahe_img, alpha)
            strategies.append(sharpened)
        
        # Стратегия 29-32: Denoising + бинаризация
        for strength in [5, 10, 15, 20]:
            denoised = cv2.fastNlMeansDenoising(gray, None, h=strength)
            _, binary_denoised = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            strategies.append(binary_denoised)
        
        # Стратегия 33-36: Морфологические операции
        kernel_sizes = [2, 3, 4, 5]
        for k_size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            
            # Закрытие для соединения разрывов
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            strategies.append(closed)
            
            # Открытие для удаления шума
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            strategies.append(opened)
        
        # Стратегия 37-40: Гамма-коррекция
        for gamma in [0.5, 0.7, 1.3, 1.5]:
            gamma_corrected = self._gamma_correction(gray, gamma)
            strategies.append(gamma_corrected)
            
            # CLAHE после гамма-коррекции
            clahe_gamma = clahe.apply(gamma_corrected)
            strategies.append(clahe_gamma)
        
        # Стратегия 41-44: Бинаризация Sauvola (через skimage если доступна)
        try:
            from skimage.filters import threshold_sauvola
            
            window_sizes = [25, 35, 45, 55]
            for w_size in window_sizes:
                if gray.shape[0] > w_size and gray.shape[1] > w_size:
                    threshold = threshold_sauvola(gray, window_size=w_size)
                    binary_sauvola = (gray > threshold).astype(np.uint8) * 255
                    strategies.append(binary_sauvola)
        except ImportError:
            pass  # skimage не установлен
        
        # Стратегия 45-48: Комбинации нескольких методов
        for i in range(4):
            # CLAHE -> Sharpen -> Adaptive Threshold
            step1 = clahe.apply(gray)
            step2 = self._sharpen_image(step1, 1.5)
            step3 = cv2.adaptiveThreshold(
                step2, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21 + i * 10,
                5
            )
            strategies.append(step3)
        
        return strategies
    
    def _sharpen_image(self, image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """
        Увеличение резкости изображения с помощью unsharp masking
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        sharpened = cv2.addWeighted(image, alpha, blurred, 1 - alpha, 0)
        return sharpened
    
    def _gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Гамма-коррекция изображения
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** (1.0 / invGamma)) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def _correct_perspective(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Коррекция перспективы для DataMatrix кода
        
        Пытается обнаружить углы кода и выпрямить изображение
        """
        # Бинаризация для поиска контуров
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Поиск контуров
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > best_area and area > 1000:  # Минимальная площадь
                # Аппроксимация полигоном
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Ищем четырёхугольники
                if len(approx) == 4:
                    best_contour = approx
                    best_area = area
        
        if best_contour is None:
            return None
        
        # Получаем точки контура
        points = best_contour.reshape(4, 2)
        
        # Сортируем точки: top-left, top-right, bottom-right, bottom-left
        rect = self._order_points(points)
        
        # Вычисляем размеры нового изображения
        width_a = np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))
        width_b = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))
        max_width = int(max(width_a, width_b))
        
        height_a = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        height_b = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        max_height = int(max(height_a, height_b))
        
        # Целевые точки
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        # Матрица перспективы
        M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Сортировка точек в порядке: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Сортировка по сумме координат (top-left имеет наименьшую сумму)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Сортировка по разности координат (top-right имеет наименьшую разность)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
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
                'avg_processing_time': 0.0,
                'cache_hits': 0,
                'parallel_decodes': 0
            }
    
    def shutdown(self):
        """Корректное завершение работы и освобождение ресурсов"""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=False)


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
