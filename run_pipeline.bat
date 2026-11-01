@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Local virtual environment not found at %~dp0.venv\Scripts\python.exe.
    echo Falling back to system python...
    set "PYTHON=python"
)

:menu
cls
echo ======================================================================
echo    Alzheimer's Disease Stage Classification ^& Grad-CAM System
echo ======================================================================
echo  1. Download Full 6,400 Real Clinical MRI Dataset
echo  2. Train AlzheimerNet Deep Attention CNN (on real data)
echo  3. Evaluate Model on Test Set (Accuracy, F1, Confusion Matrix)
echo  4. Run Single Scan Prediction ^& Grad-CAM CLI
echo  5. Launch NeuroScan Web Application Dashboard
echo  6. Generate Synthetic MRI Brain Scans (Alternative/Testing)
echo  7. Exit
echo ======================================================================
set /p choice="Select an option (1-7): "

if "%choice%"=="1" (
    echo [INFO] Downloading 6,400 clinical MRI scans...
    "%PYTHON%" download_real_dataset.py
    pause
    goto menu
)
if "%choice%"=="2" (
    echo [INFO] Starting model training...
    "%PYTHON%" train.py --epochs 10 --batch_size 32
    pause
    goto menu
)
if "%choice%"=="3" (
    echo [INFO] Evaluating model...
    "%PYTHON%" evaluate.py
    pause
    goto menu
)
if "%choice%"=="4" (
    set /p imgpath="Enter path to MRI scan image: "
    "%PYTHON%" predict.py --image "%imgpath%"
    pause
    goto menu
)
if "%choice%"=="5" (
    echo [INFO] Starting Flask web server on http://127.0.0.1:5000...
    "%PYTHON%" app\server.py
    pause
    goto menu
)
if "%choice%"=="6" (
    echo [INFO] Generating synthetic dataset...
    "%PYTHON%" sample_data_generator.py --samples 50
    pause
    goto menu
)
if "%choice%"=="7" (
    exit /b 0
)

goto menu
