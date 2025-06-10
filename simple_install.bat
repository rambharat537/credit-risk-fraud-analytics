@echo off
echo ====================================================
echo Simple Installation - Credit Risk Analytics
echo ====================================================

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install essential packages step by step
echo Installing core packages...
pip install setuptools wheel
pip install pandas numpy matplotlib seaborn
pip install scikit-learn xgboost
pip install jupyter notebook
pip install fastapi uvicorn
pip install pyyaml python-dotenv tqdm joblib

echo.
echo ✅ Core packages installed!
echo.
echo 🚀 Now you can run:
echo    python run_analysis.py
echo.
pause