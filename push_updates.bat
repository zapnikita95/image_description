@echo off
chcp 65001 >nul
cd /d "%~dp0"
set REPO_URL=https://github.com/zapnikita95/image_description.git

where git >nul 2>&1
if errorlevel 1 (
  echo [FAIL] Git not found. Install Git and add to PATH: https://git-scm.com/download/win
  pause
  exit /b 1
)

if not exist ".git" (
  echo Initializing git repo...
  git init
  git branch -M main
  git remote add origin %REPO_URL%
)

if exist ".git\index.lock" (
  echo Removing stale lock...
  del /f /q ".git\index.lock" 2>nul
)
git rm -r --cached _git_push 2>nul
git rm -r --cached data 2>nul
git add -A
git status
set /p OK="Commit and push? (y/n): "
if /i not "%OK%"=="y" exit /b 0

git commit -m "Update: training logs, torch in deps and run.bat, Ollama memory UI and warmup on save"
if errorlevel 1 (
  echo Nothing to commit or commit failed.
  pause
  exit /b 1
)

if defined GITHUB_TOKEN (
  echo Pushing with token...
  git push https://%GITHUB_TOKEN%@github.com/zapnikita95/image_description.git main
) else (
  git push -u origin main 2>nul || git push origin main
)
if errorlevel 1 (
  echo.
  echo If 403/401: set GITHUB_TOKEN=ghp_your_token then run this bat again.
  echo If remote has other commits: git pull origin main --rebase then run again.
  pause
  exit /b 1
)
echo.
echo Done. Pushed to https://github.com/zapnikita95/image_description
pause
