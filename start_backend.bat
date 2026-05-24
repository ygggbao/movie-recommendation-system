@echo off
cd /d "D:\download\movie-recommendation-system\backend"
set "PYTHON_EXE=C:\Users\LZK\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" app_v2.py
pause
