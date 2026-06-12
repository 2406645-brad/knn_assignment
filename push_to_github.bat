@echo off
title Push KNN Assignment to GitHub
echo ========================================================
echo          Pushing KNN Assignment to GitHub             
echo ========================================================
echo.
echo [*] Checking local git status...
git status
echo.
echo [*] Target Repository: https://github.com/2406645-brad/knn_assignment.git
echo.
echo [!] IMPORTANT: Please ensure you have created a repository named
echo     'knn_assignment' on your GitHub account (https://github.com/new)
echo     before proceeding!
echo.
set /p proceed="Do you want to proceed with pushing the code? (Y/N): "
if /i "%proceed%" neq "Y" (
    echo [!] Push cancelled.
    goto end
)
echo.
echo [*] Running: git push -u origin main
git push -u origin main
echo.
if %errorlevel% equ 0 (
    echo [SUCCESS] Code pushed successfully to your GitHub repository!
) else (
    echo [ERROR] Push failed. If this is a login issue, please ensure git is authenticated.
)
:end
echo.
pause
