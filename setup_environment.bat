@echo off
echo Setting up Python environment...

REM Uninstall existing pylsl if present
pip uninstall -y pylsl

REM Install specific version of pylsl known to work
pip install pylsl==1.16.1

REM Install other requirements
pip install -r requirements.txt

echo Environment setup complete!
echo Please restart the system using start_system.bat 