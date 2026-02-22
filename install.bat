@echo off
echo === Pascribe Setup ===
echo.

python -m venv venv
call venv\Scripts\activate.bat

pip install -r requirements.txt
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

echo.
echo === Done! Run with: venv\Scripts\python pascribe.py ===
pause
