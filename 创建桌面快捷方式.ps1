$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) '本地AI变声器.lnk'))
$shortcut.TargetPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$shortcut.Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + (Join-Path $PSScriptRoot '启动变声器.ps1') + '"'
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.Description = '本地 RVC 实时变声器'
$shortcut.Save()
Write-Output '已创建桌面快捷方式。'
