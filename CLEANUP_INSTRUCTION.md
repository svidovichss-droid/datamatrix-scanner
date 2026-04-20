# Решение проблемы с субмодулем realtime-capture-win

## Описание проблемы

При выполнении команды `git submodule foreach` возникает ошибка:
```
fatal: URL для пути субмодуля 'realtime-capture-win' в .gitmodules не найден
```

Это происходит, когда:
- В репозитории существует папка `realtime-capture-win`
- Git считает её субмодулем (осталась запись в конфигурации или индексе)
- Файл `.gitmodules` отсутствует или не содержит записи об этом субмодуле

## Решение

### Вариант 1: Использование скрипта очистки

#### Для Linux/macOS/Git Bash:
```bash
chmod +x cleanup_submodule.sh
./cleanup_submodule.sh
```

#### Для Windows (CMD):
```cmd
cleanup_submodule.bat
```

### Вариант 2: Ручное выполнение команд

Выполните следующие команды в корне репозитория:

```bash
# 1. Удалить секцию субмодуля из локальной конфигурации
git config --local --remove-section submodule.realtime-capture-win 2>/dev/null || true

# 2. Удалить кэш модуля (если существует)
rm -rf .git/modules/realtime-capture-win 2>/dev/null || true
# Для Windows:
# rmdir /s /q .git\modules\realtime-capture-win 2>nul

# 3. Удалить запись из индекса (если есть)
git update-index --force-remove realtime-capture-win 2>/dev/null || true

# 4. Если папка пуста - удалить её
rmdir realtime-capture-win 2>/dev/null || true
```

### Вариант 3: Для предотвращения ошибки в CI/CD скриптах

Если вы используете команду `git submodule foreach` в скриптах автоматизации, добавьте обработку ошибки:

```bash
# Проверка наличия .gitmodules перед выполнением
if [ -f .gitmodules ]; then
    git submodule foreach --recursive "ваша команда"
else
    echo "Файл .gitmodules не найден, пропускаем обработку субмодулей"
fi

# ИЛИ игнорирование ошибки
git submodule foreach --recursive "ваша команда" || true
```

## Проверка результата

После выполнения очистки проверьте статус репозитория:
```bash
git status
git submodule status
```

Ошибки больше не должно возникать.
