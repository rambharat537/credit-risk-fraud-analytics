@echo off
echo ====================================================
echo Cleaning Project for GitHub Upload
echo ====================================================

echo 🗑️ Removing large files and folders...

REM Remove virtual environment (largest folder)
if exist "venv" (
    echo Removing virtual environment...
    rmdir /s /q venv
)

REM Remove generated data files
if exist "data\sample\*.csv" (
    echo Removing sample CSV files...
    del /q data\sample\*.csv
)

REM Remove trained models
if exist "models\*.pkl" (
    echo Removing model files...
    del /q models\*.pkl
)

if exist "models\*.joblib" (
    del /q models\*.joblib
)

REM Remove logs
if exist "logs" (
    echo Removing logs...
    rmdir /s /q logs
)

REM Remove Jupyter checkpoints
if exist ".ipynb_checkpoints" (
    rmdir /s /q .ipynb_checkpoints
)

REM Remove Python cache
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

REM Remove dashboard exports
if exist "dashboards\exports\*.png" (
    echo Removing dashboard exports...
    del /q dashboards\exports\*.png
)

if exist "dashboards\exports\*.json" (
    del /q dashboards\exports\*.json
)

echo.
echo ✅ Cleanup completed!
echo.

REM Show current folder size
echo 📊 Checking folder size...
powershell -Command "& {$size = (Get-ChildItem -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB; Write-Host \"Current folder size: $([math]::Round($size, 2)) MB\"}"

echo.
echo 🚀 Ready for GitHub upload!
echo.
pause