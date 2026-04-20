# API документация

## Модули приложения

### quality_analyzer

```python
from src.quality_analyzer import DataMatrixQualityAnalyzer, detect_datamatrix_region, decode_datamatrix
```

#### DataMatrixQualityAnalyzer

```python
analyzer = DataMatrixQualityAnalyzer()
```

**Методы:**

##### analyze(image: np.ndarray, decode_result: Optional[str] = None) -> DataMatrixMetrics

Анализ качества DataMatrix кода.

**Параметры:**
- `image` — Изображение DataMatrix кода (numpy array)
- `decode_result` — Расшифрованные данные (опционально)

**Возвращает:** `DataMatrixMetrics` с результатами анализа

##### get_recommendations(metrics: DataMatrixMetrics) -> list

Генерация рекомендаций по улучшению качества.

---

#### DataMatrixMetrics

```python
@dataclass
class DataMatrixMetrics:
    rmax: float              # Макс. неравномерность (0-100%)
    contrast: float          # Контраст (0-1)
    ane: float              # Ошибка неравномерности
    cell_integrity: float   # Целостность модулей (0-100%)
    edge_snr: float         # SNR краёв
    decode_success: bool    # Успешность декодирования
    data_content: str       # Расшифрованные данные
    symbol_size: Tuple[int, int]  # Размер символа
    modules_count: int       # Количество модулей
    overall_grade: PrintQualityGrade  # Итоговая оценка
    grade_score: float       # Баллы (0-100)
```

#### detect_datamatrix_region(image: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]

Обнаружение области DataMatrix на изображении.

**Параметры:**
- `image` — Входное изображение

**Возвращает:** Кортеж (обрезанное_изображение, bbox) или None

---

### database

```python
from src.database import ScanHistoryDB, get_database
```

#### ScanHistoryDB

```python
db = ScanHistoryDB()  # Автоматически создаёт БД
# или
db = ScanHistoryDB("путь/к/базе.db")
```

**Методы:**

##### add_scan(scan_data: Dict) -> int

Добавление записи о сканировании.

```python
scan_id = db.add_scan({
    'data_content': '01:04601234567890:21:ABC123',
    'overall_grade': 'A (Отлично)',
    'grade_score': 95.5,
    'contrast': 0.85,
    'rmax': 35.2,
    'ane': 15.3,
    'cell_integrity': 98.1,
    'edge_snr': 45.2,
    'decode_success': True,
})
```

##### get_scans(limit: int = 100, offset: int = 0, ...) -> List[Dict]

Получение записей истории.

```python
scans = db.get_scans(
    limit=50,
    offset=0,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    grade_filter='F'  # Только неудачные
)
```

##### get_statistics(start_date: datetime = None, end_date: datetime = None) -> Dict

Получение статистики.

```python
stats = db.get_statistics()
# {
#     'total_scans': 1500,
#     'grade_distribution': {'A': 800, 'B': 400, 'C': 200, 'D': 80, 'F': 20},
#     'avg_metrics': {'grade_score': 78.5, 'contrast': 0.72, ...},
#     'decode_success_rate': 98.5
# }
```

##### export_to_csv(filepath: str, start_date: datetime = None, end_date: datetime = None) -> int

Экспорт в CSV файл.

```python
count = db.export_to_csv('report.csv', start_date=last_month)
```

---

### camera

```python
from src.camera import CameraManager, ImageAcquisition, CameraStatus
```

#### CameraManager

```python
camera = CameraManager()
```

**Методы:**

##### connect(device_id: int = 0) -> bool

Подключение к камере.

```python
if camera.connect(0):  # USB камера 0
    print("Подключено")
```

##### disconnect()

Отключение от камеры.

##### start_streaming(callback: Callable[[np.ndarray], None] = None) -> bool

Запуск потоковой трансляции.

```python
def on_frame(frame):
    # Обработка кадра
    pass

camera.start_streaming(callback=on_frame)
```

##### capture_frame() -> Optional[np.ndarray]

Захват одиночного кадра.

##### get_available_cameras() -> list

Поиск доступных камер.

```python
cameras = camera.get_available_cameras()
# [0, 1, 2]
```

---

#### ImageAcquisition

```python
acq = ImageAcquisition(camera)
```

**Методы:**

##### capture_and_preprocess() -> Optional[np.ndarray]

Захват и предобработка изображения.

##### preprocess_image(image: np.ndarray) -> np.ndarray

Предобработка для улучшения распознавания:
- CLAHE усиление контраста
- Шумоподавление
- Повышение резкости

---

## Использование в коде

### Пример: Полный цикл анализа

```python
from src.camera import CameraManager
from src.quality_analyzer import DataMatrixQualityAnalyzer, detect_datamatrix_region, decode_datamatrix
from src.database import get_database

# Инициализация
camera = CameraManager()
analyzer = DataMatrixQualityAnalyzer()
db = get_database()

# Подключение к камере
camera.connect(0)

# Захват кадра
frame = camera.capture_frame()

# Обнаружение и анализ
result = detect_datamatrix_region(frame)
if result:
    roi, bbox = result
    decoded = decode_datamatrix(roi)
    metrics = analyzer.analyze(roi, decoded)

    # Сохранение в историю
    db.add_scan({
        'data_content': decoded or '',
        'overall_grade': metrics.overall_grade.value,
        'grade_score': metrics.grade_score,
        'contrast': metrics.contrast,
        'rmax': metrics.rmax,
        'ane': metrics.ane,
        'cell_integrity': metrics.cell_integrity,
        'edge_snr': metrics.edge_snr,
        'decode_success': metrics.decode_success,
    })

    # Рекомендации
    for rec in analyzer.get_recommendations(metrics):
        print(rec)
```

---

## Конфигурация

### Параметры оценки

```python
analyzer = DataMatrixQualityAnalyzer()

# Пороговые значения для оценок
analyzer.grade_thresholds = {
    PrintQualityGrade.A: 90,
    PrintQualityGrade.B: 70,
    PrintQualityGrade.C: 50,
    PrintQualityGrade.D: 30,
}

# Минимальные требования
analyzer.min_requirements = {
    'contrast': 0.7,
    'rmax': 50,
    'cell_integrity': 80,
    'ane': 30,
}
```

### Конфигурация камеры

```python
from src.camera import CameraConfig

config = CameraConfig(
    device_id=0,
    resolution=(1920, 1080),
    fps=30,
    brightness=128,
    contrast=128,
    exposure=100,
    auto_exposure=False
)

camera = CameraManager()
camera.config = config
```
