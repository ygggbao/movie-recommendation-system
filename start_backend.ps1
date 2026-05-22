$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
Set-Location "D:\download\movie-recommendation-system\backend"
python app.py
Read-Host "Press Enter to exit"
