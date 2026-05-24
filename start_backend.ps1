$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
Set-Location "D:\download\movie-recommendation-system\backend"
$pythonExe = "C:\Users\LZK\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}
& $pythonExe app_v2.py
Read-Host "Press Enter to exit"
