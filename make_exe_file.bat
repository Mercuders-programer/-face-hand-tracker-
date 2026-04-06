@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================
echo   PoseTracker - Windows EXE Build Script
echo ================================================
echo.

cd /d "%~dp0"

:: Output path: x64\YYYY-MM-DD_HH-mm-ss
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'"') do set TIMESTAMP=%%t

set OUT_BASE=x64
set OUT_DIR=%OUT_BASE%\%TIMESTAMP%

if not exist "%OUT_BASE%" mkdir "%OUT_BASE%"
mkdir "%OUT_DIR%"

echo   Output: %OUT_DIR%
echo.

:: STEP 1. Python
echo [STEP 1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo   %%v
echo.

:: STEP 2. Packages
echo [STEP 2/6] Checking packages...

python -c "import mediapipe" >nul 2>&1
if errorlevel 1 ( echo   Installing mediapipe... & pip install mediapipe )

python -c "import cv2" >nul 2>&1
if errorlevel 1 ( echo   Installing opencv-python... & pip install opencv-python )

python -c "import PIL" >nul 2>&1
if errorlevel 1 ( echo   Installing Pillow... & pip install Pillow )

python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo   Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 ( echo [ERROR] PyInstaller install failed & pause & exit /b 1 )
)
echo   All packages OK
echo.

:: STEP 3. Model files
echo [STEP 3/6] Checking model files...
if not exist "models\face_landmarker.task" (
    echo [ERROR] models\face_landmarker.task not found
    pause & exit /b 1
)
if not exist "models\hand_landmarker.task" (
    echo [ERROR] models\hand_landmarker.task not found
    pause & exit /b 1
)
echo   face_landmarker.task  OK
echo   hand_landmarker.task  OK
echo.

:: STEP 4. Source files
echo [STEP 4/6] Checking source files...
set MISSING=0
if not exist "app.py"              ( echo [ERROR] app.py missing              & set MISSING=1 )
if not exist "src\camera_panel.py" ( echo [ERROR] src\camera_panel.py missing & set MISSING=1 )
if not exist "src\video_panel.py"  ( echo [ERROR] src\video_panel.py missing  & set MISSING=1 )
if not exist "src\tracker.py"      ( echo [ERROR] src\tracker.py missing      & set MISSING=1 )
if not exist "src\exporter.py"     ( echo [ERROR] src\exporter.py missing     & set MISSING=1 )
if !MISSING! == 1 ( echo [ERROR] Missing source files & pause & exit /b 1 )
echo   All source files OK
echo.

:: STEP 5. Clean previous build
echo [STEP 5/6] Cleaning previous build...
if exist "build"            rmdir /s /q "build"
if exist "dist"             rmdir /s /q "dist"
if exist "PoseTracker.spec" del /q "PoseTracker.spec"
echo   Done
echo.

:: STEP 6. PyInstaller build
echo [STEP 6/6] Building EXE (may take several minutes)...
echo.

python -m PyInstaller --name "PoseTracker" --onedir --windowed --distpath "%OUT_DIR%" --add-data "models;models" --add-data "src;src" --collect-all "mediapipe" --hidden-import "PIL" --hidden-import "PIL.Image" --hidden-import "PIL.ImageTk" --hidden-import "cv2" app.py

if errorlevel 1 (
    echo.
    echo ================================================
    echo   [BUILD FAILED] Check the error above.
    echo ================================================
    pause & exit /b 1
)

if exist "build"            rmdir /s /q "build"
if exist "PoseTracker.spec" del /q "PoseTracker.spec"

echo.
echo ================================================
echo   Build complete!
echo   EXE: %OUT_DIR%\PoseTracker\PoseTracker.exe
echo ================================================
echo.
pause
