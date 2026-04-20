@echo off
REM Скрипт для очистки проблемного субмодуля realtime-capture-win (Windows)
REM Запускать из корня репозитория

echo === Очистка проблемного субмодуля realtime-capture-win ===

REM 1. Удалить секцию субмодуля из локальной конфигурации (если есть)
echo Проверка конфигурации субмодулей...
git config --local --get-regexp "submodule\.realtime-capture-win\." >nul 2>&1
if %errorlevel% equ 0 (
    echo Удаление записи о субмодуле из конфигурации...
    git config --local --remove-section submodule.realtime-capture-win
) else (
    echo Записей о субмодуле в конфигурации не найдено
)

REM 2. Удалить кэш модуля (если существует)
if exist ".git\modules\realtime-capture-win" (
    echo Удаление кэша модуля .git\modules\realtime-capture-win...
    rmdir /s /q ".git\modules\realtime-capture-win"
) else (
    echo Кэш модуля не найден
)

REM 3. Удалить запись из индекса (если есть)
echo Проверка индекса на наличие субмодуля...
git ls-files --stage | findstr "^160000.*realtime-capture-win" >nul 2>&1
if %errorlevel% equ 0 (
    echo Удаление записи субмодуля из индекса...
    git update-index --force-remove realtime-capture-win
) else (
    echo Записи субмодуля в индексе не найдено
)

REM 4. Если папка пуста и не нужна - удалить её
if exist "realtime-capture-win\" (
    dir /b "realtime-capture-win" >nul 2>&1
    if %errorlevel% neq 0 (
        echo Папка realtime-capture-win пуста, удаление...
        rmdir /s /q "realtime-capture-win"
    ) else (
        echo Внимание: Папка realtime-capture-win содержит файлы, оставляем без изменений
    )
)

REM 5. Проверить статус репозитория
echo.
echo === Статус репозитория ===
git status

echo.
echo === Очистка завершена ===
pause
