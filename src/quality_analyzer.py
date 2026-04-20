"""
Модуль анализа качества печати DataMatrix по ГОСТ Р 57302-2016

Авторы: А. Свидович / А. Петляков для PROGRESS
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from enum import Enum
import math


class PrintQualityGrade(Enum):
    """Оценка качества печати по ГОСТ Р 57302-2016"""
    A = "A (Отлично)"
    B = "B (Хорошо)"
    C = "C (Удовлетворительно)"
    D = "D (Плохо)"
    F = "F (Непригоден)"


@dataclass
class DataMatrixMetrics:
    """Метрики качества DataMatrix кода"""
    # Основные метрики по ГОСТ Р 57302-2016
    rmax: float = 0.0          # Максимальная неравномерность (0-100%)
    contrast: float = 0.0      # Контраст (мин/макс яркость)
    ane: float = 0.0           # Averaged Non-Uniformity Error
    cell_integrity: float = 0.0  # Целостность ячеек (0-100%)
    edge_snr: float = 0.0      # Signal-to-Noise Ratio краёв

    # Дополнительные метрики
    decode_success: bool = False
    data_content: str = ""
    symbol_size: Tuple[int, int] = (0, 0)
    modules_count: int = 0

    # Параметры для итоговой оценки
    overall_grade: PrintQualityGrade = PrintQualityGrade.F
    grade_score: float = 0.0    # 0-100 баллов


class DataMatrixQualityAnalyzer:
    """
    Анализатор качества печати DataMatrix кодов

    Реализует методику оценки по ГОСТ Р 57302-2016:
    - Анализ контраста
    - Оценка неравномерности яркости
    - Проверка целостности модулей
    - Расчёт SNR краёв
    """

    def __init__(self):
        # Пороговые значения для разных оценок
        self.grade_thresholds = {
            PrintQualityGrade.A: 90,
            PrintQualityGrade.B: 70,
            PrintQualityGrade.C: 50,
            PrintQualityGrade.D: 30,
        }

        # Требования к минимальным значениям
        self.min_requirements = {
            'contrast': 0.7,      # Минимальный контраст
            'rmax': 50,           # Максимальная неравномерность (%)
            'cell_integrity': 80, # Минимальная целостность (%)
            'ane': 30,            # Максимальная ошибка неравномерности
        }

    def analyze(self, image: np.ndarray, decode_result: Optional[str] = None) -> DataMatrixMetrics:
        """
        Полный анализ качества DataMatrix кода

        Args:
            image: Изображение DataMatrix кода (обрезанное)
            decode_result: Расшифрованные данные (опционально)

        Returns:
            DataMatrixMetrics с результатами анализа
        """
        metrics = DataMatrixMetrics()

        if image is None or image.size == 0:
            return metrics

        try:
            # Подготовка изображения
            gray = self._prepare_image(image)

            # Расчёт контраста
            metrics.contrast = self._calculate_contrast(gray)

            # Расчёт Rmax (максимальная неравномерность)
            metrics.rmax = self._calculate_rmax(gray)

            # Расчёт ANE (Averaged Non-Uniformity Error)
            metrics.ane = self._calculate_ane(gray)

            # Оценка целостности модулей
            metrics.cell_integrity = self._calculate_cell_integrity(gray)

            # Расчёт SNR краёв
            metrics.edge_snr = self._calculate_edge_snr(gray)

            # Итоговая оценка
            metrics.overall_grade, metrics.grade_score = self._calculate_overall_grade(metrics)

            # Дополнительная информация
            if decode_result:
                metrics.decode_success = True
                metrics.data_content = decode_result

            # Размер и количество модулей
            metrics.symbol_size = (image.shape[1], image.shape[0])
            metrics.modules_count = self._estimate_modules_count(image)

        except Exception as e:
            print(f"Ошибка анализа: {e}")

        return metrics

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Подготовка изображения к анализу"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Убеждаемся, что изображение квадратное
        h, w = gray.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = gray

        return square

    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """
        Расчёт контраста DataMatrix кода

        Контраст = (Max - Min) / (Max + Min)
        По ГОСТ требуется контраст >= 0.7
        """
        # Находим яркости тёмных и светлых модулей
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Находим пики яркости для светлых и тёмных модулей
        # DataMatrix: обычно светлый фон и тёмные модули
        threshold = np.median(gray)

        dark_pixels = gray[gray < threshold]
        light_pixels = gray[gray >= threshold]

        if len(dark_pixels) == 0 or len(light_pixels) == 0:
            return 0.0

        dark_mean = np.mean(dark_pixels)
        light_mean = np.mean(light_pixels)

        # Контраст по формуле Михеля
        contrast = (light_mean - dark_mean) / 255.0

        return min(1.0, max(0.0, contrast))

    def _calculate_rmax(self, gray: np.ndarray) -> float:
        """
        Расчёт Rmax - максимальной неравномерности яркости

        Rmax = 100 * (Gmax - Gmin) / Gср
        где Gmax, Gmin - максимальная и минимальная яркость
        Gср - средняя яркость
        """
        # Разбиваем на регионы
        h, w = gray.shape
        grid_size = 5  # Размер сетки для анализа

        region_h = h // grid_size
        region_w = w // grid_size

        region_means = []

        for i in range(grid_size):
            for j in range(grid_size):
                region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                if region.size > 0:
                    region_means.append(np.mean(region))

        if not region_means:
            return 100.0

        region_means = np.array(region_means)
        g_max = np.max(region_means)
        g_min = np.min(region_means)
        g_avg = np.mean(region_means)

        if g_avg == 0:
            return 100.0

        rmax = 100.0 * (g_max - g_min) / g_avg

        return min(100.0, rmax)

    def _calculate_ane(self, gray: np.ndarray) -> float:
        """
        Расчёт ANE - Average Non-Uniformity Error

        ANE характеризует отклонение яркости модулей от идеальных значений
        """
        # Определяем порог для разделения на модули
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Оценка размера модуля
        symbol_size = min(gray.shape)
        module_size = symbol_size // 10  # Примерная оценка

        if module_size < 2:
            return 100.0

        # Подсчёт неравномерности
        errors = []

        # Анализ горизонтальных линий
        for y in range(module_size, gray.shape[0] - module_size, module_size):
            row = gray[y, :]
            expected = np.mean(row)
            variance = np.var(row)
            errors.append(variance)

        # Анализ вертикальных линий
        for x in range(module_size, gray.shape[1] - module_size, module_size):
            col = gray[:, x]
            variance = np.var(col)
            errors.append(variance)

        if not errors:
            return 100.0

        # ANE как среднее отклонение
        ane = np.mean(errors) / 255.0 * 100

        return min(100.0, ane)

    def _calculate_cell_integrity(self, gray: np.ndarray) -> float:
        """
        Оценка целостности модулей (cell integrity)

        Проверяет, насколько модули соответствуют идеальной сетке
        """
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Поиск контуров
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Находим основной контур (сам код)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Оценка заполненности
        roi = binary[y:y+h, x:x+w]
        fill_ratio = np.sum(roi == 0) / (w * h)  # Тёмные модули

        # Идеальное соотношение для DataMatrix ~50-60%
        # Отклонение от этого показывает целостность
        ideal_fill = 0.55
        integrity = 100.0 * (1.0 - abs(fill_ratio - ideal_fill) / ideal_fill)

        return min(100.0, max(0.0, integrity))

    def _calculate_edge_snr(self, gray: np.ndarray) -> float:
        """
        Расчёт SNR (Signal-to-Noise Ratio) краёв модулей

        Высокий SNR указывает на чёткие края модулей
        """
        # Обнаружение краёв
        edges = cv2.Canny(gray, 50, 150)

        # Сигнал: пиксели краёв
        signal = np.sum(edges > 0)

        # Шум: дисперсия в областях без краёв
        noise_regions = gray[edges == 0]
        noise = np.std(noise_regions) if len(noise_regions) > 0 else 1

        if noise == 0:
            return 100.0

        snr = signal / (noise * 1000)

        return min(100.0, snr * 10)

    def _estimate_modules_count(self, image: np.ndarray) -> int:
        """Оценка количества модулей в DataMatrix"""
        h, w = image.shape
        # Типичный размер DataMatrix: от 10x10 до 144x144
        estimated = min(h, w) // 4  # Примерная оценка
        return max(10, min(144, estimated))

    def _calculate_overall_grade(self, metrics: DataMatrixMetrics) -> Tuple[PrintQualityGrade, float]:
        """
        Расчёт итоговой оценки качества

        Формула учитывает все метрики с весовыми коэффициентами
        """
        # Весовые коэффициенты метрик
        weights = {
            'contrast': 0.25,
            'rmax': 0.25,
            'cell_integrity': 0.25,
            'ane': 0.15,
            'edge_snr': 0.10,
        }

        # Нормализация метрик (0-100)
        def normalize(value: float, metric: str) -> float:
            req = self.min_requirements.get(metric, 50)
            if req == 0:
                return 100.0
            # Для некоторых метрик меньше = лучше
            if metric in ['rmax', 'ane']:
                return max(0, min(100, 100 - value))
            return max(0, min(100, value))

        scores = {
            'contrast': normalize(metrics.contrast * 100, 'contrast'),
            'rmax': normalize(metrics.rmax, 'rmax'),
            'cell_integrity': metrics.cell_integrity,
            'ane': normalize(metrics.ane, 'ane'),
            'edge_snr': metrics.edge_snr,
        }

        # Расчёт взвешенной суммы
        total_score = sum(scores[k] * weights[k] for k in weights)

        # Определение оценки
        grade = PrintQualityGrade.F
        for g, threshold in sorted(self.grade_thresholds.items(), key=lambda x: -x[1]):
            if total_score >= threshold:
                grade = g
                break

        return grade, round(total_score, 1)

    def get_recommendations(self, metrics: DataMatrixMetrics) -> list:
        """
        Генерация рекомендаций по улучшению качества печати
        """
        recommendations = []

        if metrics.contrast < self.min_requirements['contrast']:
            recommendations.append(
                f"Увеличить контраст (текущий: {metrics.contrast:.2f}, "
                f"требуется: >= {self.min_requirements['contrast']:.2f})"
            )

        if metrics.rmax > self.min_requirements['rmax']:
            recommendations.append(
                f"Устранить неравномерность печати (Rmax: {metrics.rmax:.1f}%, "
                f"требуется: <= {self.min_requirements['rmax']}%)"
            )

        if metrics.cell_integrity < self.min_requirements['cell_integrity']:
            recommendations.append(
                f"Проверить целостность модулей (текущая: {metrics.cell_integrity:.1f}%, "
                f"требуется: >= {self.min_requirements['cell_integrity']}%)"
            )

        if metrics.ane > self.min_requirements['ane']:
            recommendations.append(
                f"Улучшить однородность яркости (ANE: {metrics.ane:.1f})"
            )

        if metrics.edge_snr < 30:
            recommendations.append(
                "Проверить резкость печати и качество материала"
            )

        if not recommendations:
            recommendations.append("Качество печати соответствует требованиям ГОСТ")

        return recommendations


def decode_datamatrix(image: np.ndarray) -> Optional[str]:
    """
    Декодирование DataMatrix кода из изображения

    Использует pyzbar для декодирования
    """
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        decoded_objects = pyzbar_decode(gray, symbols=[2])  # 2 = DataMatrix

        if decoded_objects:
            return decoded_objects[0].data.decode('utf-8')

    except ImportError:
        print("Внимание: pyzbar не установлен, декодирование недоступно")
    except Exception as e:
        print(f"Ошибка декодирования: {e}")

    return None


def detect_datamatrix_region(image: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Обнаружение области DataMatrix на изображении

    Returns:
        Кортеж (обрезанное_изображение, (x, y, w, h))
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Усиление контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Обнаружение краёв
    edges = cv2.Canny(enhanced, 50, 150)

    # Поиск квадратных контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_score = 0

    for contour in contours:
        # Аппроксимация контура
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Ищем квадратные/прямоугольные объекты
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            # Проверяем, что объект квадратный (допуск 30%)
            aspect_ratio = w / h if h > 0 else 0
            if 0.7 <= aspect_ratio <= 1.3:
                # Проверяем размер (должен быть достаточно большим)
                area = w * h
                img_area = image.shape[0] * image.shape[1]

                if area > img_area * 0.001 and area < img_area * 0.5:
                    # Оценка качества кандидата
                    roi = enhanced[y:y+h, x:x+w]
                    contrast = np.std(roi)

                    if contrast > best_score:
                        best_score = contrast
                        best_candidate = (x, y, w, h)

    if best_candidate:
        x, y, w, h = best_candidate
        # Добавляем отступ
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        roi = image[y1:y2, x1:x2]
        return roi, (x1, y1, x2 - x1, y2 - y1)

    return None
