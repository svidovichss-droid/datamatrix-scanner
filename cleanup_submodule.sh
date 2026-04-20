#!/bin/bash
# Скрипт для очистки проблемного субмодуля realtime-capture-win
# Запускать из корня репозитория

set -e

echo "=== Очистка проблемного субмодуля realtime-capture-win ==="

# 1. Удалить секцию субмодуля из локальной конфигурации (если есть)
echo "Проверка конфигурации субмодулей..."
if git config --local --get-regexp "submodule\.realtime-capture-win\." >/dev/null 2>&1; then
    echo "Удаление записи о субмодуле из конфигурации..."
    git config --local --remove-section submodule.realtime-capture-win || true
else
    echo "Записей о субмодуле в конфигурации не найдено"
fi

# 2. Удалить кэш модуля (если существует)
if [ -d ".git/modules/realtime-capture-win" ]; then
    echo "Удаление кэша модуля .git/modules/realtime-capture-win..."
    rm -rf .git/modules/realtime-capture-win
else
    echo "Кэш модуля не найден"
fi

# 3. Удалить запись из индекса (если есть)
echo "Проверка индекса на наличие субмодуля..."
if git ls-files --stage | grep "^160000.*realtime-capture-win" >/dev/null 2>&1; then
    echo "Удаление записи субмодуля из индекса..."
    git update-index --force-remove realtime-capture-win
else
    echo "Записи субмодуля в индексе не найдено"
fi

# 4. Если папка пуста и не нужна - удалить её
if [ -d "realtime-capture-win" ]; then
    if [ -z "$(ls -A realtime-capture-win 2>/dev/null)" ]; then
        echo "Папка realtime-capture-win пуста, удаление..."
        rm -rf realtime-capture-win
    else
        echo "Внимание: Папка realtime-capture-win содержит файлы, оставляем без изменений"
    fi
fi

# 5. Проверить статус репозитория
echo ""
echo "=== Статус репозитория ==="
git status

echo ""
echo "=== Очистка завершена ==="
