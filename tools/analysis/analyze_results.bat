@echo off
setlocal
REM Usage: analyze_results.bat [RESULTS] [ALGO] [BEST] [OUT]
REM Defaults: results, bstsp, (none), results\summary\results_summary.csv
set ROOT=%~dp0\..
cd /d "%ROOT%"

if not exist analyze_results.py (
  echo ERROR: analyze_results.py not found in %ROOT%
  exit /b 1
)

set RESULTS=%1
if "%RESULTS%"=="" set RESULTS=results
set ALGO=%2
if "%ALGO%"=="" set ALGO=bstsp
set BEST=%3
set OUT=%4
if "%OUT%"=="" set OUT=results\summary\results_summary.csv

if not exist "%RESULTS%" (
  echo ERROR: Results directory not found: %RESULTS%
  exit /b 1
)

for %%F in ("%OUT%") do if not exist "%%~dpF" mkdir "%%~dpF"

set CMD=python analyze_results.py --results "%RESULTS%" --algo "%ALGO%" --out "%OUT%"
if not "%BEST%"=="" if exist "%BEST%" set CMD=%CMD% --best "%BEST%"
%CMD%
