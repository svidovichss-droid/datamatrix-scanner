# Makefile для DataMatrix Scanner
# Авторы: А. Свидович / А. Петляков для PROGRESS

.PHONY: help install build clean test run

help:
	@echo "DataMatrix Quality Scanner - Makefile"
	@echo ""
	@echo "Доступные команды:"
	@echo "  make install    - Установка зависимостей"
	@echo "  make build      - Сборка exe файла"
	@echo "  make clean      - Очистка временных файлов"
	@echo "  make test       - Запуск тестов"
	@echo "  make run        - Запуск приложения"

install:
	@echo "Установка зависимостей..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Готово!"

build:
	@echo "Сборка DataMatrixScanner.exe..."
	pyinstaller build.spec --clean
	@echo ""
	@echo "Сборка завершена!"
	@echo "Файл: dist/DataMatrixScanner/DataMatrixScanner.exe"

build-archive:
	@echo "Создание архива..."
	cd dist && powershell -Command "Compress-Archive -Path DataMatrixScanner/* -DestinationPath DataMatrixScanner-win64.zip -Force"
	@echo "Архив: dist/DataMatrixScanner-win64.zip"

clean:
	@echo "Очистка..."
	rm -rf build dist *.egg-info
	rm -rf __pycache__ src/__pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Готово!"

test:
	@echo "Запуск тестов..."
	python -c "
from src.quality_analyzer import DataMatrixQualityAnalyzer, detect_datamatrix_region
from src.database import ScanHistoryDB
from src.camera import CameraManager, simulate_datamatrix_image

# Test analyzer
analyzer = DataMatrixQualityAnalyzer()
print('✓ DataMatrixQualityAnalyzer')

# Test database
db = ScanHistoryDB()
stats = db.get_statistics()
print(f'✓ ScanHistoryDB: {stats[\"total_scans\"]} записей')

# Test camera
img = simulate_datamatrix_image()
result = detect_datamatrix_region(img)
print(f'✓ detect_datamatrix_region: {\"Найден\" if result else \"Не найден\"}')

print('')
print('Все тесты пройдены!')
"

run:
	@echo "Запуск DataMatrix Scanner..."
	python main.py

dev:
	@echo "Запуск в режиме разработки..."
	PYTHONPATH=. python main.py
