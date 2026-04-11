@echo off
cd /d "%~dp0"
set ZIP=feed_image_attributes.zip
if exist "%ZIP%" del "%ZIP%"
powershell -NoProfile -Command "Compress-Archive -Path 'run.py','feed_parser.py','ollama_vision.py','requirements.txt','README.md','run.command','run.bat','run.sh','make_archive.sh','make_archive.command','make_archive.bat' -DestinationPath '%ZIP%' -Force"
if exist "%ZIP%" (
  echo Created: %cd%\%ZIP%
  echo Transfer this file to another computer, unzip, then follow README.md
) else (
  echo Failed to create archive. Ensure PowerShell is available.
)
pause
