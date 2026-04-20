# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
        # OpenCV DLLs
    ],
    datas=[
        # Ресурсы приложения
    ],
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'cv2',
        'numpy',
        'sqlalchemy',
        'pyzbar',
        'pylibdmtx',
        'loguru',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'PIL',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DataMatrixScanner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Без консольного окна
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='x64',  # Только x64
    codesign_identity='',
    entitlements_file=None,
    icon='assets/icon.ico',  # Иконка приложения
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    target_arch='x64',
    name='DataMatrixScanner',
)

# Создание установочного архива
dist = DIST(
    coll,
    name='DataMatrixScanner-Setup',
    exclude=[],
)
