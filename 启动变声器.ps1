$ErrorActionPreference = 'Stop'
try {
    $configPath = Join-Path $PSScriptRoot 'runtime.json'
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw '请复制 runtime.example.json 为 runtime.json，并填写 engine_dir 和 python_path。'
    }
    $runtime = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $pythonPath = [string]$runtime.python_path
    if (-not [IO.Path]::IsPathRooted($pythonPath)) {
        $pythonPath = Join-Path $PSScriptRoot $pythonPath
    }
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw 'runtime.json 中的 python_path 不存在，请选择已安装 RVC 依赖的 Python 环境。'
    }
    $app = Join-Path $PSScriptRoot 'voice_changer.py'
    Start-Process -FilePath $pythonPath -ArgumentList ('"' + $app + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, '本地 AI 变声器') | Out-Null
}
