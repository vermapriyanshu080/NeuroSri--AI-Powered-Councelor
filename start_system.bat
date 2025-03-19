@echo off
echo Starting EEG Analysis System...

REM Start CHORDS LSL Stream in a separate window
start "CHORDS LSL Stream" cmd /k "python src/chords/chords.py -p COM3 -b 115200 --lsl"

REM Wait for CHORDS stream to initialize
timeout /t 3

REM Start the Python backend server
start cmd /k "python server.py"

REM Wait for backend to initialize
timeout /t 5

REM Start the frontend development server
cd frontend
start cmd /k "npm run dev"

REM open the frontend in the browser
start http://localhost:5173

REM 
cd..

echo System started successfully!
echo CHORDS LSL Stream running in separate window
echo Backend running on http://localhost:5000
echo Frontend running on http://localhost:5173 