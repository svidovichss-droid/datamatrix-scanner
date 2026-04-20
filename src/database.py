"""
Модуль базы данных для хранения истории сканирований

Авторы: А. Свидович / А. Петляков для PROGRESS
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import asdict
from pathlib import Path
import os


class ScanHistoryDB:
    """
    База данных истории сканирований

    Структура таблицы scans:
    - id: INTEGER PRIMARY KEY
    - timestamp: DATETIME
    - data_content: TEXT (содержимое DataMatrix)
    - overall_grade: TEXT (A, B, C, D, F)
    - grade_score: REAL (0-100)
    - contrast: REAL
    - rmax: REAL
    - ane: REAL
    - cell_integrity: REAL
    - edge_snr: REAL
    - symbol_size: TEXT (w,h)
    - image_path: TEXT (путь к сохранённому изображению)
    - recommendations: TEXT (JSON массив)
    - camera_id: TEXT (ID камеры)
    - line_speed: REAL (скорость конвейера, м/мин)
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Создаём в папке с приложением
            app_dir = Path(__file__).parent.parent
            db_dir = app_dir / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "scan_history.db"

        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        """Инициализация структуры базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    data_content TEXT,
                    overall_grade TEXT,
                    grade_score REAL,
                    contrast REAL,
                    rmax REAL,
                    ane REAL,
                    cell_integrity REAL,
                    edge_snr REAL,
                    symbol_size TEXT,
                    image_path TEXT,
                    recommendations TEXT,
                    camera_id TEXT,
                    line_speed REAL,
                    decode_success INTEGER DEFAULT 0
                )
            """)

            # Индексы для быстрого поиска
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON scans(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_grade
                ON scans(overall_grade)
            """)

            conn.commit()

    def add_scan(self, scan_data: Dict) -> int:
        """
        Добавление записи о сканировании

        Args:
            scan_data: Словарь с данными сканирования

        Returns:
            ID новой записи
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO scans (
                    data_content, overall_grade, grade_score,
                    contrast, rmax, ane, cell_integrity, edge_snr,
                    symbol_size, image_path, recommendations,
                    camera_id, line_speed, decode_success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scan_data.get('data_content', ''),
                scan_data.get('overall_grade', 'F'),
                scan_data.get('grade_score', 0),
                scan_data.get('contrast', 0),
                scan_data.get('rmax', 0),
                scan_data.get('ane', 0),
                scan_data.get('cell_integrity', 0),
                scan_data.get('edge_snr', 0),
                scan_data.get('symbol_size', ''),
                scan_data.get('image_path', ''),
                json.dumps(scan_data.get('recommendations', []), ensure_ascii=False),
                scan_data.get('camera_id', ''),
                scan_data.get('line_speed', 0),
                1 if scan_data.get('decode_success', False) else 0
            ))

            conn.commit()
            return cursor.lastrowid

    def get_scans(self, limit: int = 100, offset: int = 0,
                  start_date: datetime = None, end_date: datetime = None,
                  grade_filter: str = None) -> List[Dict]:
        """
        Получение записей истории сканирований

        Args:
            limit: Максимальное количество записей
            offset: Смещение для пагинации
            start_date: Начало периода фильтрации
            end_date: Конец периода фильтрации
            grade_filter: Фильтр по оценке (A, B, C, D, F)

        Returns:
            Список записей
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM scans WHERE 1=1"
            params = []

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            if grade_filter:
                query += " AND overall_grade = ?"
                params.append(grade_filter)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_statistics(self, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """
        Получение статистики сканирований

        Returns:
            Словарь со статистикой
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            base_query = ""
            params = []

            if start_date or end_date:
                base_query = " WHERE 1=1"
                if start_date:
                    base_query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                if end_date:
                    base_query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())

            # Общее количество
            cursor.execute(f"SELECT COUNT(*) FROM scans{base_query}", params)
            total_count = cursor.fetchone()[0]

            # Количество по оценкам
            cursor.execute(f"""
                SELECT overall_grade, COUNT(*) as count
                FROM scans{base_query}
                GROUP BY overall_grade
            """, params)
            grade_distribution = {row[0]: row[1] for row in cursor.fetchall()}

            # Средние метрики
            cursor.execute(f"""
                SELECT
                    AVG(grade_score) as avg_score,
                    AVG(contrast) as avg_contrast,
                    AVG(rmax) as avg_rmax,
                    AVG(cell_integrity) as avg_integrity
                FROM scans{base_query}
            """, params)
            row = cursor.fetchone()
            avg_metrics = {
                'grade_score': row[0] or 0,
                'contrast': row[1] or 0,
                'rmax': row[2] or 0,
                'cell_integrity': row[3] or 0
            }

            # Процент успешного декодирования
            cursor.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(decode_success) as successful
                FROM scans{base_query}
            """, params)
            row = cursor.fetchone()
            decode_rate = (row[1] / row[0] * 100) if row[0] > 0 else 0

            return {
                'total_scans': total_count,
                'grade_distribution': grade_distribution,
                'avg_metrics': avg_metrics,
                'decode_success_rate': round(decode_rate, 1)
            }

    def delete_scan(self, scan_id: int) -> bool:
        """Удаление записи сканирования"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
            conn.commit()
            return cursor.rowcount > 0

    def export_to_csv(self, filepath: str, start_date: datetime = None, end_date: datetime = None):
        """Экспорт истории в CSV файл"""
        import csv

        scans = self.get_scans(limit=100000, start_date=start_date, end_date=end_date)

        if not scans:
            return 0

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Получаем все ключи из первой записи
            fieldnames = list(scans[0].keys())

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(scans)

        return len(scans)

    def clear_old_records(self, days: int = 90) -> int:
        """
        Удаление старых записей

        Args:
            days: Удалить записи старше указанного количества дней

        Returns:
            Количество удалённых записей
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM scans
                WHERE timestamp < datetime('now', ? || ' days')
            """, (-days,))

            conn.commit()
            return cursor.rowcount

    def get_recent_failures(self, limit: int = 10) -> List[Dict]:
        """Получение последних неудачных сканирований"""
        return self.get_scans(
            limit=limit,
            grade_filter='F'
        )


# Глобальный экземпляр базы данных
_db_instance = None


def get_database() -> ScanHistoryDB:
    """Получение экземпляра базы данных"""
    global _db_instance
    if _db_instance is None:
        _db_instance = ScanHistoryDB()
    return _db_instance
