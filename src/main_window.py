"""
Графический интерфейс сканера DataMatrix

Авторы: А. Свидович / А. Петляков для PROGRESS
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QGroupBox, QFrame, QStatusBar, QMenuBar, QMenu, QToolBar,
    QDial, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QTabWidget, QTextEdit, QProgressBar, QSplitter, QFileDialog,
    QMessageBox, QDialog, QCheckBox, QDateEdit, QListWidget,
    QListWidgetItem, QScrollArea, QSizePolicy, QStyleFactory
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QDate, QDateTime,
    QSettings, QUrl
)
from PyQt6.QtGui import (
    QImage, QPixmap, QIcon, QAction, QColor, QPalette,
    QPainter, QPen, QBrush, QFont
)

import cv2

from .camera import CameraManager, CameraStatus, ImageAcquisition, simulate_datamatrix_image
from .quality_analyzer import DataMatrixQualityAnalyzer, DataMatrixMetrics, PrintQualityGrade, detect_datamatrix_region, decode_datamatrix
from .database import ScanHistoryDB, get_database


class WorkerThread(QThread):
    """Поток для обработки изображений"""
    frame_processed = pyqtSignal(np.ndarray, object)  # frame, metrics
    scan_completed = pyqtSignal(object)  # metrics
    error_occurred = pyqtSignal(str)

    def __init__(self, analyzer: DataMatrixQualityAnalyzer):
        super().__init__()
        self.analyzer = analyzer
        self.frame = None
        self.running = False

    def process_frame(self, frame: np.ndarray):
        """Обработка кадра"""
        self.frame = frame

    def run(self):
        """Обработка в отдельном потоке"""
        self.running = True

        while self.running and self.frame is not None:
            try:
                frame = self.frame.copy()

                # Детекция области DataMatrix
                result = detect_datamatrix_region(frame)

                if result:
                    roi, bbox = result

                    # Рисуем рамку вокруг обнаруженного кода
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Декодирование
                    decoded = decode_datamatrix(roi)

                    # Анализ качества
                    metrics = self.analyzer.analyze(roi, decoded)

                    self.frame_processed.emit(frame, metrics)

                    if metrics.decode_success:
                        self.scan_completed.emit(metrics)
                else:
                    self.frame_processed.emit(frame, None)

            except Exception as e:
                self.error_occurred.emit(str(e))

            QThread.msleep(50)  # Ограничение частоты обработки

    def stop(self):
        """Остановка потока"""
        self.running = False


class QualityGauge(QWidget):
    """Виджет отображения качества в виде полукруглого индикатора"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.max_value = 100
        self.grade = PrintQualityGrade.F
        self.setMinimumSize(150, 100)

    def setValue(self, value: float, grade: PrintQualityGrade = None):
        self.value = value
        if grade:
            self.grade = grade
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        center = (w // 2, h - 20)
        radius = min(w, h) - 40

        # Цвета для разных оценок
        colors = {
            PrintQualityGrade.A: QColor(0, 200, 0),     # Зелёный
            PrintQualityGrade.B: QColor(150, 200, 0),   # Жёлто-зелёный
            PrintQualityGrade.C: QColor(255, 200, 0),   # Жёлтый
            PrintQualityGrade.D: QColor(255, 100, 0),   # Оранжевый
            PrintQualityGrade.F: QColor(255, 0, 0),     # Красный
        }

        # Фоновая дуга
        painter.setPen(QPen(QColor(50, 50, 50), 15))
        painter.drawArc(
            center[0] - radius, center[1] - radius,
            radius * 2, radius * 2,
            180 * 16, 180 * 16
        )

        # Цветная дуга значения
        color = colors.get(self.grade, QColor(128, 128, 128))
        painter.setPen(QPen(color, 15))

        angle = int((self.value / self.max_value) * 180 * 16)
        painter.drawArc(
            center[0] - radius, center[1] - radius,
            radius * 2, radius * 2,
            180 * 16, -angle  # Отрицательный угол для отсчёта от 180°
        )

        # Текст значения
        painter.setPen(color)
        font = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            center[0] - 30, center[1] - 10,
            f"{self.value:.0f}"
        )

        # Текст оценки
        font = QFont("Arial", 14)
        painter.setFont(font)
        painter.drawText(
            center[0] - 15, center[1] + 20,
            self.grade.value.split()[0] if hasattr(self.grade, 'value') else "N/A"
        )


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()

        # Компоненты системы
        self.camera = CameraManager()
        self.analyzer = DataMatrixQualityAnalyzer()
        self.db = get_database()
        self.image_acq = ImageAcquisition(self.camera)
        self.worker = WorkerThread(self.analyzer)

        # Состояние
        self.current_metrics: DataMatrixMetrics = None
        self.auto_capture = False
        self.captured_images = []

        # Загрузка настроек
        self.settings = QSettings("PROGRESS", "DataMatrixScanner")

        # Инициализация UI
        self.init_ui()

        # Таймеры
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_preview)

        # Подключение сигналов
        self.worker.frame_processed.connect(self.on_frame_processed)
        self.worker.scan_completed.connect(self.on_scan_completed)
        self.worker.error_occurred.connect(self.on_error)

        # Восстановление настроек
        self.restore_settings()

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("Сканер качества печати DataMatrix - PROGRESS")
        self.setMinimumSize(1200, 800)

        # Меню
        self.create_menu()

        # Центральный виджет
        central = QWidget()
        self.setCentralWidget(central)

        # Основной layout
        main_layout = QHBoxLayout(central)

        # Левая панель - видео и управление
        left_panel = self.create_left_panel()

        # Центральная панель - результаты
        center_panel = self.create_center_panel()

        # Правая панель - история
        right_panel = self.create_right_panel()

        # Сплиттеры
        splitter1 = QSplitter(Qt.Orientation.Horizontal)
        splitter1.addWidget(left_panel)
        splitter1.addWidget(center_panel)
        splitter1.addWidget(right_panel)
        splitter1.setStretchFactor(0, 3)
        splitter1.setStretchFactor(1, 2)
        splitter1.setStretchFactor(2, 2)

        main_layout.addWidget(splitter1)

        # Статус бар
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_menu(self):
        """Создание меню"""
        menubar = self.menuBar()

        # Файл
        file_menu = menubar.addMenu("&Файл")

        export_action = QAction("&Экспорт истории...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_history)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        settings_action = QAction("&Настройки...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QAction("&Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Камера
        camera_menu = menubar.addMenu("&Камера")

        connect_action = QAction("&Подключить...", self)
        connect_action.setShortcut("Ctrl+K")
        connect_action.triggered.connect(self.connect_camera)
        camera_menu.addAction(connect_action)

        disconnect_action = QAction("&Отключить", self)
        disconnect_action.triggered.connect(self.disconnect_camera)
        camera_menu.addAction(disconnect_action)

        camera_menu.addSeparator()

        test_image_action = QAction("&Тестовое изображение", self)
        test_image_action.triggered.connect(self.show_test_image)
        camera_menu.addAction(test_image_action)

        # Справка
        help_menu = menubar.addMenu("&Справка")

        about_action = QAction("&О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        gost_action = QAction("&ГОСТ Р 57302-2016", self)
        gost_action.triggered.connect(self.show_gost_info)
        help_menu.addAction(gost_action)

    def create_left_panel(self) -> QWidget:
        """Создание левой панели (видео и управление)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Группа видео
        video_group = QGroupBox("Видеопоток")
        video_layout = QVBoxLayout(video_group)

        # Метка для отображения видео
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setFrameStyle(QFrame.Shape.Box)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Камера не подключена")
        self.video_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        video_layout.addWidget(self.video_label)

        layout.addWidget(video_group)

        # Группа управления камерой
        ctrl_group = QGroupBox("Управление")
        ctrl_layout = QGridLayout(ctrl_group)

        # Кнопки управления
        self.connect_btn = QPushButton("Подключить камеру")
        self.connect_btn.clicked.connect(self.connect_camera)
        ctrl_layout.addWidget(self.connect_btn, 0, 0)

        self.capture_btn = QPushButton("Захватить кадр")
        self.capture_btn.clicked.connect(self.capture_frame)
        self.capture_btn.setEnabled(False)
        ctrl_layout.addWidget(self.capture_btn, 0, 1)

        self.scan_btn = QPushButton("Начать сканирование")
        self.scan_btn.clicked.connect(self.toggle_scanning)
        self.scan_btn.setEnabled(False)
        ctrl_layout.addWidget(self.scan_btn, 1, 0, 1, 2)

        # Автозахват
        self.auto_capture_cb = QCheckBox("Автозахват при обнаружении")
        self.auto_capture_cb.toggled.connect(lambda c: setattr(self, 'auto_capture', c))
        ctrl_layout.addWidget(self.auto_capture_cb, 2, 0, 1, 2)

        layout.addWidget(ctrl_group)

        # Параметры камеры
        params_group = QGroupBox("Параметры камеры")
        params_layout = QGridLayout(params_group)

        params_layout.addWidget(QLabel("Яркость:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(0, 255)
        self.brightness_slider.setValue(128)
        self.brightness_slider.valueChanged.connect(
            lambda v: self.camera.update_config(brightness=v)
        )
        params_layout.addWidget(self.brightness_slider, 0, 1)

        params_layout.addWidget(QLabel("Контраст:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 255)
        self.contrast_slider.setValue(128)
        self.contrast_slider.valueChanged.connect(
            lambda v: self.camera.update_config(contrast=v)
        )
        params_layout.addWidget(self.contrast_slider, 1, 1)

        params_layout.addWidget(QLabel("Экспозиция:"), 2, 0)
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(1, 1000)
        self.exposure_slider.setValue(100)
        self.exposure_slider.valueChanged.connect(
            lambda v: self.camera.update_config(exposure=v)
        )
        params_layout.addWidget(self.exposure_slider, 2, 1)

        layout.addWidget(params_group)

        return panel

    def create_center_panel(self) -> QWidget:
        """Создание центральной панели (результаты)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Вкладки результатов
        tabs = QTabWidget()

        # Вкладка основных метрик
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)

        # Индикатор качества
        gauge_layout = QHBoxLayout()
        gauge_layout.addStretch()

        self.quality_gauge = QualityGauge()
        gauge_layout.addWidget(self.quality_gauge)
        gauge_layout.addStretch()

        metrics_layout.addLayout(gauge_layout)

        # Таблица метрик
        metrics_group = QGroupBox("Метрики качества (ГОСТ Р 57302-2016)")
        metrics_grid = QGridLayout(metrics_group)

        self.metrics_labels = {}

        metrics_data = [
            ("Контраст:", "contrast"),
            ("Rmax (%):", "rmax"),
            ("ANE:", "ane"),
            ("Целостность (%):", "cell_integrity"),
            ("SNR краёв:", "edge_snr"),
        ]

        for row, (label, key) in enumerate(metrics_data):
            metrics_grid.addWidget(QLabel(label), row, 0)
            value_label = QLabel("—")
            value_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            metrics_grid.addWidget(value_label, row, 1)
            self.metrics_labels[key] = value_label

        metrics_grid.addWidget(QLabel("Размер символа:"), 5, 0)
        self.size_label = QLabel("—")
        metrics_grid.addWidget(self.size_label, 5, 1)

        metrics_grid.addWidget(QLabel("Модулей:"), 6, 0)
        self.modules_label = QLabel("—")
        metrics_grid.addWidget(self.modules_label, 6, 1)

        metrics_layout.addWidget(metrics_group)

        # Данные DataMatrix
        data_group = QGroupBox("Данные DataMatrix")
        data_layout = QVBoxLayout(data_group)

        self.data_content_label = QLabel("—")
        self.data_content_label.setWordWrap(True)
        self.data_content_label.setStyleSheet(
            "background-color: #f5f5f5; padding: 8px; border-radius: 4px;"
        )
        data_layout.addWidget(self.data_content_label)

        self.decode_status_label = QLabel("Декодирование: —")
        data_layout.addWidget(self.decode_status_label)

        metrics_layout.addWidget(data_group)

        # Рекомендации
        rec_group = QGroupBox("Рекомендации")
        rec_layout = QVBoxLayout(rec_group)

        self.recommendations_list = QListWidget()
        rec_layout.addWidget(self.recommendations_list)

        metrics_layout.addWidget(rec_group)

        tabs.addTab(metrics_tab, "Метрики")

        # Вкладка статистики
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        refresh_btn = QPushButton("Обновить статистику")
        refresh_btn.clicked.connect(self.update_statistics)
        stats_layout.addWidget(refresh_btn)

        tabs.addTab(stats_tab, "Статистика")

        layout.addWidget(tabs)

        return panel

    def create_right_panel(self) -> QWidget:
        """Создание правой панели (история)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Фильтры
        filter_group = QGroupBox("Фильтры")
        filter_layout = QGridLayout(filter_group)

        filter_layout.addWidget(QLabel("Дата с:"), 0, 0)
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addMonths(-1))
        self.date_from.setCalendarPopup(True)
        filter_layout.addWidget(self.date_from, 0, 1)

        filter_layout.addWidget(QLabel("Оценка:"), 1, 0)
        self.grade_filter = QComboBox()
        self.grade_filter.addItems(["Все", "A", "B", "C", "D", "F"])
        filter_layout.addWidget(self.grade_filter, 1, 1)

        apply_btn = QPushButton("Применить")
        apply_btn.clicked.connect(self.load_history)
        filter_layout.addWidget(apply_btn, 2, 0, 1, 2)

        layout.addWidget(filter_group)

        # Таблица истории
        history_group = QGroupBox("История сканирований")
        history_layout = QVBoxLayout(history_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Время", "Данные", "Оценка", "Баллы", "Статус"
        ])
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.cellClicked.connect(self.on_history_row_clicked)
        self.history_table.setMaximumHeight(300)
        history_layout.addWidget(self.history_table)

        # Кнопки работы с историей
        history_btns = QHBoxLayout()

        refresh_btn = QPushButton("Обновить")
        refresh_btn.clicked.connect(self.load_history)
        history_btns.addWidget(refresh_btn)

        export_btn = QPushButton("Экспорт")
        export_btn.clicked.connect(self.export_history)
        history_btns.addWidget(export_btn)

        delete_btn = QPushButton("Удалить")
        delete_btn.clicked.connect(self.delete_selected_scan)
        history_btns.addWidget(delete_btn)

        history_layout.addLayout(history_btns)

        layout.addWidget(history_group)

        # Детали выбранного сканирования
        detail_group = QGroupBox("Детали")
        detail_layout = QVBoxLayout(detail_group)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(150)
        detail_layout.addWidget(self.detail_text)

        layout.addWidget(detail_group)

        return panel

    def connect_camera(self):
        """Подключение к камере"""
        # Получаем список доступных камер
        cameras = self.camera.get_available_cameras()

        if not cameras:
            QMessageBox.information(
                self,
                "Камера не найдена",
                "USB камеры не обнаружены.\n"
                "Подключите камеру или используйте тестовое изображение "
                "(Меню: Камера > Тестовое изображение)"
            )
            return

        # Подключаемся к первой доступной камере
        if self.camera.connect(cameras[0]):
            self.status_bar.showMessage(f"Камера подключена (ID: {cameras[0]})")
            self.connect_btn.setText("Отключить камеру")
            self.connect_btn.clicked.disconnect()
            self.connect_btn.clicked.connect(self.disconnect_camera)

            self.capture_btn.setEnabled(True)
            self.scan_btn.setEnabled(True)

            self.frame_timer.start(33)  # ~30 FPS
        else:
            QMessageBox.warning(
                self,
                "Ошибка подключения",
                f"Не удалось подключиться к камере:\n{self.camera.get_last_error()}"
            )

    def disconnect_camera(self):
        """Отключение от камеры"""
        self.frame_timer.stop()
        self.worker.stop()
        self.camera.disconnect()

        self.status_bar.showMessage("Камера отключена")
        self.connect_btn.setText("Подключить камеру")
        self.connect_btn.clicked.disconnect()
        self.connect_btn.clicked.connect(self.connect_camera)

        self.capture_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)

        self.video_label.clear()
        self.video_label.setText("Камера не подключена")
        self.video_label.setStyleSheet("background-color: #1a1a1a; color: #888;")

    def show_test_image(self):
        """Показать тестовое изображение"""
        # Создаём тестовое изображение
        test_img = simulate_datamatrix_image(400)
        self.display_frame(test_img)

        # Обрабатываем
        self.worker.process_frame(test_img)
        self.worker.start()

    def toggle_scanning(self):
        """Переключение режима сканирования"""
        if self.camera.is_streaming:
            self.worker.stop()
            self.scan_btn.setText("Начать сканирование")
            self.status_bar.showMessage("Сканирование остановлено")
        else:
            if self.worker.start():
                self.scan_btn.setText("Остановить сканирование")
                self.status_bar.showMessage("Сканирование запущено...")

    def capture_frame(self):
        """Захват текущего кадра"""
        frame = self.camera.capture_frame()
        if frame is not None:
            self.display_frame(frame)
            self.worker.process_frame(frame)
            self.worker.start()

    def update_preview(self):
        """Обновление превью"""
        frame = self.camera.current_frame
        if frame is not None:
            self.display_frame(frame)

    def display_frame(self, frame: np.ndarray):
        """Отображение кадра"""
        # Конвертация BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Создание QImage
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Масштабирование
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def on_frame_processed(self, frame: np.ndarray, metrics: DataMatrixMetrics):
        """Обработка результата анализа кадра"""
        self.display_frame(frame)

        if metrics:
            self.update_metrics_display(metrics)

    def update_metrics_display(self, metrics: DataMatrixMetrics):
        """Обновление отображения метрик"""
        self.current_metrics = metrics

        # Обновление индикатора
        self.quality_gauge.setValue(metrics.grade_score, metrics.overall_grade)

        # Обновление метрик
        self.metrics_labels["contrast"].setText(f"{metrics.contrast:.2f}")
        self.metrics_labels["rmax"].setText(f"{metrics.rmax:.1f}%")
        self.metrics_labels["ane"].setText(f"{metrics.ane:.1f}")
        self.metrics_labels["cell_integrity"].setText(f"{metrics.cell_integrity:.1f}%")
        self.metrics_labels["edge_snr"].setText(f"{metrics.edge_snr:.1f}")

        # Размер символа
        if metrics.symbol_size[0] > 0:
            self.size_label.setText(f"{metrics.symbol_size[0]}x{metrics.symbol_size[1]}")
            self.modules_label.setText(str(metrics.modules_count))

        # Данные
        if metrics.decode_success:
            self.data_content_label.setText(metrics.data_content)
            self.decode_status_label.setText("Декодирование: Успешно")
            self.decode_status_label.setStyleSheet("color: green;")
        else:
            self.data_content_label.setText("(не удалось декодировать)")
            self.decode_status_label.setText("Декодирование: Ошибка")
            self.decode_status_label.setStyleSheet("color: red;")

        # Рекомендации
        self.recommendations_list.clear()
        recommendations = self.analyzer.get_recommendations(metrics)
        for rec in recommendations:
            item = QListWidgetItem(f"• {rec}")
            if "соответствует" in rec:
                item.setForeground(QColor(0, 150, 0))
            else:
                item.setForeground(QColor(255, 100, 0))
            self.recommendations_list.addItem(item)

    def on_scan_completed(self, metrics: DataMatrixMetrics):
        """Обработка завершённого сканирования"""
        self.update_metrics_display(metrics)

        # Автосохранение в историю
        if self.auto_capture and metrics.decode_success:
            self.save_scan_to_history(metrics)

        self.status_bar.showMessage(
            f"Сканирование завершено: {metrics.overall_grade.value} ({metrics.grade_score:.1f} баллов)"
        )

    def save_scan_to_history(self, metrics: DataMatrixMetrics):
        """Сохранение сканирования в историю"""
        scan_data = {
            'data_content': metrics.data_content,
            'overall_grade': metrics.overall_grade.value,
            'grade_score': metrics.grade_score,
            'contrast': metrics.contrast,
            'rmax': metrics.rmax,
            'ane': metrics.ane,
            'cell_integrity': metrics.cell_integrity,
            'edge_snr': metrics.edge_snr,
            'symbol_size': f"{metrics.symbol_size[0]},{metrics.symbol_size[1]}",
            'recommendations': self.analyzer.get_recommendations(metrics),
            'decode_success': metrics.decode_success,
        }

        scan_id = self.db.add_scan(scan_data)
        self.load_history()

        return scan_id

    def load_history(self):
        """Загрузка истории сканирований"""
        # Получение параметров фильтра
        date_from = self.date_from.dateTime().toPyDateTime()
        grade_filter = self.grade_filter.currentText()

        # Загрузка данных
        scans = self.db.get_scans(
            limit=100,
            start_date=date_from,
            grade_filter=None if grade_filter == "Все" else grade_filter
        )

        # Заполнение таблицы
        self.history_table.setRowCount(len(scans))

        for row, scan in enumerate(scans):
            # Время
            timestamp = scan['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            self.history_table.setItem(row, 0, QTableWidgetItem(
                timestamp.strftime("%d.%m.%Y %H:%M:%S")
            ))

            # Данные (обрезаем длинные)
            data = scan['data_content'][:30] + "..." if len(scan['data_content']) > 30 else scan['data_content']
            self.history_table.setItem(row, 1, QTableWidgetItem(data))

            # Оценка
            grade_item = QTableWidgetItem(scan['overall_grade'])
            grade_colors = {
                'A (Отлично)': QColor(0, 150, 0),
                'B (Хорошо)': QColor(100, 180, 0),
                'C (Удовлетворительно)': QColor(255, 200, 0),
                'D (Плохо)': QColor(255, 150, 0),
                'F (Непригоден)': QColor(255, 0, 0),
            }
            grade_item.setForeground(grade_colors.get(scan['overall_grade'], QColor(0, 0, 0)))
            self.history_table.setItem(row, 2, grade_item)

            # Баллы
            self.history_table.setItem(row, 3, QTableWidgetItem(
                f"{scan['grade_score']:.1f}"
            ))

            # Статус декодирования
            status = "OK" if scan['decode_success'] else "FAIL"
            status_item = QTableWidgetItem(status)
            status_item.setForeground(QColor(0, 150, 0) if scan['decode_success'] else QColor(255, 0, 0))
            self.history_table.setItem(row, 4, status_item)

        # Обновление статистики
        self.update_statistics()

    def on_history_row_clicked(self, row: int):
        """Обработка клика по строке истории"""
        # Здесь можно добавить отображение деталей
        pass

    def delete_selected_scan(self):
        """Удаление выбранного сканирования"""
        current_row = self.history_table.currentRow()
        if current_row < 0:
            return

        # Получаем ID из данных
        item = self.history_table.item(current_row, 0)
        if not item:
            return

        reply = QMessageBox.question(
            self,
            "Удаление",
            "Удалить выбранную запись?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Здесь нужно получить ID записи из БД
            self.load_history()

    def update_statistics(self):
        """Обновление статистики"""
        date_from = self.date_from.dateTime().toPyDateTime()
        stats = self.db.get_statistics(start_date=date_from)

        text = f"""
<h2>Статистика сканирований</h2>

<p><b>Всего сканирований:</b> {stats['total_scans']}</p>

<p><b>Успешность декодирования:</b> {stats['decode_success_rate']:.1f}%</p>

<h3>Распределение по оценкам:</h3>
<table>
"""
        for grade, count in sorted(stats['grade_distribution'].items()):
            pct = count / stats['total_scans'] * 100 if stats['total_scans'] > 0 else 0
            text += f"<tr><td>{grade}</td><td>{count} ({pct:.1f}%)</td></tr>"

        text += "</table>"

        text += f"""
<h3>Средние метрики:</h3>
<ul>
<li>Общий балл: {stats['avg_metrics']['grade_score']:.1f}</li>
<li>Контраст: {stats['avg_metrics']['contrast']:.2f}</li>
<li>Rmax: {stats['avg_metrics']['rmax']:.1f}%</li>
<li>Целостность: {stats['avg_metrics']['cell_integrity']:.1f}%</li>
</ul>
"""

        self.stats_text.setHtml(text)

    def export_history(self):
        """Экспорт истории в CSV"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт истории",
            f"datamatrix_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV файлы (*.csv)"
        )

        if filename:
            date_from = self.date_from.dateTime().toPyDateTime()
            count = self.db.export_to_csv(filename, start_date=date_from)
            QMessageBox.information(
                self,
                "Экспорт",
                f"Экспортировано {count} записей в {filename}"
            )

    def show_settings(self):
        """Показать настройки"""
        QMessageBox.information(
            self,
            "Настройки",
            "Настройки приложения будут доступны в следующей версии.\n\n"
            "Текущие параметры:\n"
            f"- База данных: {self.db.db_path}\n"
            f"- Состояние камеры: {self.camera.status.value}"
        )

    def show_about(self):
        """О программе"""
        QMessageBox.about(
            self,
            "О программе",
            "<h3>Сканер качества печати DataMatrix</h3>"
            "<p>Версия 1.0.0</p>"
            "<p><b>Авторы:</b><br>А. Свидович / А. Петляков</p>"
            "<p><b>Компания:</b> PROGRESS</p>"
            "<hr>"
            "<p>Программа для оценки качества печати DataMatrix кодов "
            "согласно ГОСТ Р 57302-2016</p>"
            "<p>Используется для контроля качества маркировки "
            "на производственных линиях.</p>"
        )

    def show_gost_info(self):
        """Информация о ГОСТ"""
        QMessageBox.information(
            self,
            "ГОСТ Р 57302-2016",
            "<h3>ГОСТ Р 57302-2016</h3>"
            "<p><b>Название:</b><br>"
            "Автоматическая идентификация. "
            "Кодирование штриховое. "
            "Матричные символики. "
            "Требования к испытаниям качества</p>"
            "<hr>"
            "<p><b>Основные параметры оценки:</b></p>"
            "<ul>"
            "<li><b>Rmax</b> - максимальная неравномерность яркости</li>"
            "<li><b>Контраст</b> - разница между светлыми и тёмными модулями</li>"
            "<li><b>ANE</b> - ошибка неравномерности</li>"
            "<li><b>Целостность</b> - сохранность структуры модулей</li>"
            "<li><b>SNR</b> - отношение сигнал/шум краёв</li>"
            "</ul>"
            "<p><b>Шкала оценок:</b></p>"
            "<ul>"
            "<li>A (90-100) - Отлично</li>"
            "<li>B (70-89) - Хорошо</li>"
            "<li>C (50-69) - Удовлетворительно</li>"
            "<li>D (30-49) - Плохо</li>"
            "<li>F (0-29) - Непригоден</li>"
            "</ul>"
        )

    def on_error(self, error_msg: str):
        """Обработка ошибок"""
        self.status_bar.showMessage(f"Ошибка: {error_msg}", 5000)

    def restore_settings(self):
        """Восстановление настроек"""
        # Позиция окна
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Загрузка истории
        self.load_history()

    def save_settings(self):
        """Сохранение настроек"""
        self.settings.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event):
        """Закрытие приложения"""
        self.save_settings()

        # Остановка всех процессов
        self.worker.stop()
        self.frame_timer.stop()
        self.camera.disconnect()

        event.accept()


def main():
    """Точка входа"""
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))

    # Применение стиля
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 5px 15px;
            min-width: 80px;
        }
        QTableWidget {
            gridline-color: #d0d0d0;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
