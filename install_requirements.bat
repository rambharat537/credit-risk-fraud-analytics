@echo off
echo ====================================================
echo Credit Risk & Fraud Analytics - Setup Script
echo ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is installed
python --version

REM Create virtual environment
echo.
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📥 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing required packages...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directory structure...
mkdir logs 2>nul
mkdir models 2>nul
mkdir dashboards\exports 2>nul

echo.
echo ✅ Setup completed successfully!
echo.
echo 🚀 To run the analysis:
echo    1. Activate virtual environment: venv\Scripts\activate.bat
echo    2. Run analysis: python run_analysis.py
echo    3. Start API server: python src\api\fraud_detection_api.py
echo.
echo 📊 Check the following after running:
echo    - Generated data: data\sample\
echo    - Trained models: models\
echo    - Analysis results: dashboards\exports\
echo.
pause