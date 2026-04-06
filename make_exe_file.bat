@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ================================================
echo   PoseTracker - Windows EXE 빌드 스크립트
echo ================================================
echo.

:: 배치파일 위치를 작업 디렉토리로 설정
cd /d "%~dp0"

:: ------------------------------------------------
:: 출력 경로 설정: x64\YYYY-MM-DD_HH-MM-SS
:: ------------------------------------------------
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format \"yyyy-MM-dd_HH-mm-ss\""') do set TIMESTAMP=%%t

set OUT_BASE=x64
set OUT_DIR=%OUT_BASE%\%TIMESTAMP%

if not exist "%OUT_BASE%" mkdir "%OUT_BASE%"
mkdir "%OUT_DIR%"

echo   출력 경로: %OUT_DIR%
echo.

:: ------------------------------------------------
:: STEP 1. Python 설치 확인
:: ------------------------------------------------
echo [STEP 1/6] Python 설치 확인...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [오류] Python을 찾을 수 없습니다.
    echo        https://python.org 에서 설치 후 PATH에 추가하세요.
    echo        설치 시 "Add Python to PATH" 체크 필수
    echo.
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo        %%v 확인 완료
echo.

:: ------------------------------------------------
:: STEP 2. 필수 패키지 확인 / 설치
:: ------------------------------------------------
echo [STEP 2/6] 필수 패키지 확인...

python -c "import mediapipe" >nul 2>&1
if errorlevel 1 (
    echo        mediapipe 설치 중...
    pip install mediapipe
)

python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo        opencv-python 설치 중...
    pip install opencv-python
)

python -c "import PIL" >nul 2>&1
if errorlevel 1 (
    echo        Pillow 설치 중...
    pip install Pillow
)

python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo        PyInstaller 설치 중...
    pip install pyinstaller
    if errorlevel 1 (
        echo [오류] PyInstaller 설치 실패
        pause & exit /b 1
    )
)
echo        모든 패키지 확인 완료
echo.

:: ------------------------------------------------
:: STEP 3. 모델 파일 확인
:: ------------------------------------------------
echo [STEP 3/6] 모델 파일 확인...

if not exist "models\face_landmarker.task" (
    echo.
    echo [오류] models\face_landmarker.task 가 없습니다.
    echo        models 폴더에 모델 파일을 넣어주세요.
    echo.
    pause & exit /b 1
)
if not exist "models\hand_landmarker.task" (
    echo.
    echo [오류] models\hand_landmarker.task 가 없습니다.
    echo        models 폴더에 모델 파일을 넣어주세요.
    echo.
    pause & exit /b 1
)
echo        face_landmarker.task  OK
echo        hand_landmarker.task  OK
echo.

:: ------------------------------------------------
:: STEP 4. 소스 파일 확인
:: ------------------------------------------------
echo [STEP 4/6] 소스 파일 확인...

set MISSING=0
if not exist "app.py"                 ( echo [오류] app.py 없음              & set MISSING=1 )
if not exist "src\camera_panel.py"    ( echo [오류] src\camera_panel.py 없음  & set MISSING=1 )
if not exist "src\video_panel.py"     ( echo [오류] src\video_panel.py 없음   & set MISSING=1 )
if not exist "src\tracker.py"         ( echo [오류] src\tracker.py 없음       & set MISSING=1 )
if not exist "src\exporter.py"        ( echo [오류] src\exporter.py 없음      & set MISSING=1 )

if !MISSING! == 1 (
    echo.
    echo [오류] 필수 소스 파일이 없습니다. 프로젝트 폴더를 확인하세요.
    pause & exit /b 1
)
echo        app.py / src/*.py  모두 확인 완료
echo.

:: ------------------------------------------------
:: STEP 5. 이전 임시 빌드 정리
:: ------------------------------------------------
echo [STEP 5/6] 임시 빌드 파일 정리...
if exist "build"            rmdir /s /q "build"
if exist "dist"             rmdir /s /q "dist"
if exist "PoseTracker.spec" del /q "PoseTracker.spec"
echo        정리 완료
echo.

:: ------------------------------------------------
:: STEP 6. PyInstaller 빌드
:: ------------------------------------------------
echo [STEP 6/6] EXE 빌드 시작...
echo        (mediapipe 포함으로 수 분 소요될 수 있습니다)
echo.

python -m PyInstaller ^
    --name "PoseTracker" ^
    --onedir ^
    --windowed ^
    --distpath "%OUT_DIR%" ^
    --add-data "models;models" ^
    --add-data "src;src" ^
    --collect-all "mediapipe" ^
    --hidden-import "PIL" ^
    --hidden-import "PIL.Image" ^
    --hidden-import "PIL.ImageTk" ^
    --hidden-import "cv2" ^
    app.py

if errorlevel 1 (
    echo.
    echo ================================================
    echo   [빌드 실패] 위 오류 메시지를 확인하세요.
    echo ================================================
    pause & exit /b 1
)

:: 임시 빌드 폴더 정리
if exist "build"            rmdir /s /q "build"
if exist "PoseTracker.spec" del /q "PoseTracker.spec"

:: ------------------------------------------------
:: 빌드 완료
:: ------------------------------------------------
echo.
echo ================================================
echo   빌드 완료!
echo.
echo   실행 파일: %OUT_DIR%\PoseTracker\PoseTracker.exe
echo.
echo   배포 시 해당 폴더 전체를 복사하세요.
echo ================================================
echo.
pause
