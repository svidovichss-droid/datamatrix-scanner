#!/bin/bash
# Скрипт для первоначальной настройки GitHub репозитория
# Замените YOUR_USERNAME и YOUR_REPO на ваши данные

set -e

echo "=== Настройка GitHub репозитория для DataMatrix Scanner ==="

# Создание репозитория на GitHub (выполните вручную через веб-интерфейс или CLI)
# gh repo create datamatrix-scanner --public --source=. --push

# Инициализация git (если не инициализирован)
if [ ! -d .git ]; then
    git init
    git add .
    git commit -m "Initial commit: DataMatrix Quality Scanner v1.0.0"
    echo ""
    echo "=== Репозиторий инициализирован ==="
fi

echo ""
echo "=== Следующие шаги ==="
echo ""
echo "1. Создайте репозиторий на GitHub:"
echo "   https://github.com/new"
echo ""
echo "2. Подключите локальный репозиторий:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/datamatrix-scanner.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Включите GitHub Actions в настройках репозитория"
echo "   (Settings > Actions > Allow all actions)"
echo ""
echo "4. Создайте тег для релиза:"
echo "   git tag v1.0.0"
echo "   git push origin v1.0.0"
echo ""
echo "5. После сборки скачайте exe:"
echo "   https://github.com/YOUR_USERNAME/datamatrix-scanner/releases"
echo ""
echo "=== Готово ==="
