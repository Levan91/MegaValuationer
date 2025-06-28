@echo off
REM Comprehensive Data Sync Script for MegaValuationer (Windows)
REM This script adds new files, updates existing ones, and pushes to GitHub

echo 🚀 MegaValuationer Data Sync Script
echo ==================================

REM Check if we're in the right directory
if not exist "vapp.py" (
    echo ❌ Error: Please run this script from the MegaValuationer directory
    pause
    exit /b 1
)

REM Check git status
echo 📊 Checking current git status...
git status --porcelain

REM Add all changes (new files and updates)
echo.
echo 📁 Adding all data files to git...
git add Data/
git add *.xlsx 2>nul

REM Check if there are any changes to commit
git diff --cached --quiet
if %errorlevel% equ 0 (
    echo ✅ No changes detected - everything is up to date!
) else (
    echo.
    echo 📝 Changes detected:
    git diff --cached --name-only
    
    REM Get timestamp for commit message
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set timestamp=%%a %%b %%c
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set timestamp=%timestamp% %%a:%%b
    
    echo.
    echo 💾 Committing changes...
    git commit -m "Sync data files - %timestamp%"
    
    echo.
    echo 🚀 Pushing to GitHub...
    git push origin main
    if %errorlevel% equ 0 (
        echo.
        echo ✅ Successfully synced to GitHub!
        echo.
        echo 📋 Summary:
        echo    • Status: Pushed to GitHub
        echo    • Timestamp: %timestamp%
        echo.
        echo 🔄 Your Streamlit app will automatically update in 1-2 minutes
        echo 💡 You can click 'Refresh All Data' in your app to clear cache
    ) else (
        echo ❌ Error pushing to GitHub. Please check your internet connection.
        pause
        exit /b 1
    )
)

echo.
echo 🎉 Data sync complete!
echo.
echo 📱 Next steps:
echo    1. Wait 1-2 minutes for Streamlit Cloud to redeploy
echo    2. Open your app and click '🔄 Refresh All Data'
echo    3. Your new/updated data will be available!

pause 