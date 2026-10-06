# Fractal-BLT Instant Onboarding TUI (fixed)
#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Fractal-BLT // Instant Onboarding"

function Write-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "  FRACTAL-BLT  //  ZERO-ALLOCATION ONBOARDING" -ForegroundColor Cyan
    Write-Host "  NVMe -> GPU  |  NativeAOT  |  MoE Streaming" -ForegroundColor DarkCyan
    Write-Host "  -----------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""
}

function Write-Status {
    param([string]$msg, [string]$color = "White")
    Write-Host "  [$([DateTime]::Now.ToString('HH:mm:ss'))] " -NoNewline -ForegroundColor DarkGray
    Write-Host $msg -ForegroundColor $color
}

function Pause-Enter {
    Write-Host ""
    Write-Host "  Press Enter to continue..." -ForegroundColor DarkGray
    [void][System.Console]::ReadLine()
}

function Get-Choice {
    param([string[]]$Options, [string]$Prompt = "Select an option")
    Write-Host ""
    for ($i = 0; $i -lt $Options.Count; $i++) {
        Write-Host ("  [{0}]  {1}" -f ($i + 1), $Options[$i]) -ForegroundColor Yellow
    }
    Write-Host ""
    do {
        $sel = Read-Host "  $Prompt (1-$($Options.Count))"
        $num = 0
        [int]::TryParse($sel, [ref]$num) | Out-Null
    } while ($num -lt 1 -or $num -gt $Options.Count)
    return $num - 1
}

# State
$script:RepoRoot   = $PSScriptRoot
$script:ModelPath  = $env:FRACTAL_MODEL
$script:ServerPort = 5000
$script:ServerProc = $null
$script:PublishedExe = $null

$possibleExes = @(
    "$RepoRoot\FractalServe\bin\Release\net10.0\win-x64\publish\FractalServe.exe",
    "$RepoRoot\FractalServe\bin\Release\net9.0\win-x64\publish\FractalServe.exe"
)
foreach ($p in $possibleExes) {
    if (Test-Path $p) { $script:PublishedExe = $p; break }
}

function Show-SystemCheck {
    Write-Banner
    Write-Host "  SYSTEM CHECK" -ForegroundColor Green
    Write-Host ""

    try {
        $dotnet = & dotnet --version 2>$null
        Write-Status ".NET SDK          : $dotnet" "Green"
    } catch {
        Write-Status ".NET SDK          : NOT FOUND" "Red"
    }

    try {
        $smi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        Write-Status "GPU               : $smi" "Green"
    } catch {
        Write-Status "GPU / nvidia-smi  : NOT DETECTED" "Yellow"
    }

    if ($script:ModelPath -and (Test-Path $script:ModelPath)) {
        $size = [math]::Round((Get-Item $script:ModelPath).Length / 1GB, 2)
        Write-Status "FRACTAL_MODEL     : $script:ModelPath ($size GB)" "Green"
    } else {
        Write-Status "FRACTAL_MODEL     : not set" "Yellow"
    }

    if ($script:PublishedExe) {
        Write-Status "Published binary  : Found" "Green"
    } else {
        Write-Status "Published binary  : Not found" "Yellow"
    }

    Pause-Enter
}

function Set-ModelPath {
    Write-Banner
    Write-Host "  SET MODEL PATH" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Enter the absolute path to your .safetensors file." -ForegroundColor White
    Write-Host "  Example: C:\models\model.safetensors" -ForegroundColor DarkGray
    Write-Host ""

    $path = Read-Host "  Path"
    if ([string]::IsNullOrWhiteSpace($path)) { return }

    $path = $path.Trim('"')

    if (-not (Test-Path $path)) {
        Write-Status "File does not exist: $path" "Red"
        Pause-Enter
        return
    }

    $script:ModelPath = $path
    $env:FRACTAL_MODEL = $path
    [Environment]::SetEnvironmentVariable("FRACTAL_MODEL", $path, "User")

    $size = [math]::Round((Get-Item $path).Length / 1GB, 2)
    Write-Status "Model set -> $path ($size GB)" "Green"
    Pause-Enter
}

function Run-SetupWizard {
    Write-Banner
    Write-Host "  EASY SETUP WIZARD" -ForegroundColor Green
    Write-Host ""
    
    $modelsDir = Join-Path $script:RepoRoot "Models"
    if (-not (Test-Path $modelsDir)) {
        New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
    }
    
    Write-Host "  Step 1: Get Your Model" -ForegroundColor Cyan
    Write-Host "  A 'Models' folder has been created at:" -ForegroundColor White
    Write-Host "  $modelsDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Please download your model (e.g., Bonsai 27B) and place the .safetensors" -ForegroundColor White
    Write-Host "  or .gguf file into this folder." -ForegroundColor White
    
    Pause-Enter
    
    Write-Status "Scanning $modelsDir for models..." "Cyan"
    
    $modelOptions = @()
    
    # Check root for loose files
    $looseFiles = Get-ChildItem -Path $modelsDir -File -Include "*.safetensors", "*.gguf"
    if ($looseFiles.Count -eq 1) {
        $modelOptions += @{ Name = $looseFiles[0].Name; Path = $looseFiles[0].FullName }
    } elseif ($looseFiles.Count -gt 1) {
        $modelOptions += @{ Name = "Root Directory (Contains $($looseFiles.Count) model fragments)"; Path = $modelsDir }
    }
    
    # Check for subdirectories
    $subDirs = Get-ChildItem -Path $modelsDir -Directory
    foreach ($dir in $subDirs) {
        $dirFiles = Get-ChildItem -Path $dir.FullName -File -Include "*.safetensors", "*.gguf" -Recurse
        if ($dirFiles.Count -gt 0) {
            $modelOptions += @{ Name = "$($dir.Name) (Folder)"; Path = $dir.FullName }
        }
    }
    
    if ($modelOptions.Count -eq 0) {
        Write-Status "No models found in the Models folder!" "Red"
        Write-Host "  You can restart the wizard when the model is downloaded." -ForegroundColor DarkGray
        Pause-Enter
        return
    } elseif ($modelOptions.Count -eq 1) {
        $selectedPath = $modelOptions[0].Path
        Write-Status "Found model: $($modelOptions[0].Name)" "Green"
    } else {
        Write-Host ""
        Write-Host "  Multiple models/folders found. Please select one:" -ForegroundColor Cyan
        $names = $modelOptions | ForEach-Object { $_.Name }
        $idx = Get-Choice -Options $names -Prompt "Select model"
        $selectedPath = $modelOptions[$idx].Path
    }
    
    $script:ModelPath = $selectedPath
    $env:FRACTAL_MODEL = $script:ModelPath
    [Environment]::SetEnvironmentVariable("FRACTAL_MODEL", $script:ModelPath, "User")
    
    Write-Host ""
    Write-Host "  Step 2: Build NativeAOT Engine" -ForegroundColor Cyan
    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
        Write-Status ".NET SDK not found. Please install .NET 9.0 or 10.0" "Red"
        Pause-Enter
        return
    }
    
    Write-Status "Publishing FractalServe (this can take a minute)..." "Cyan"
    Push-Location $script:RepoRoot
    try {
        & dotnet publish FractalServe\FractalServe.csproj -c Release -r win-x64 /p:PublishAot=true --verbosity minimal
        if ($LASTEXITCODE -eq 0) {
            $script:PublishedExe = "$RepoRoot\FractalServe\bin\Release\net10.0\win-x64\publish\FractalServe.exe"
            if (-not (Test-Path $script:PublishedExe)) {
                $script:PublishedExe = "$RepoRoot\FractalServe\bin\Release\net9.0\win-x64\publish\FractalServe.exe"
            }
            Write-Status "Build succeeded" "Green"
        } else {
            Write-Status "Build failed. Server will run using 'dotnet run' instead." "Yellow"
        }
    } finally {
        Pop-Location
    }
    
    Write-Host ""
    Write-Host "  Step 3: Start Server" -ForegroundColor Cyan
    
    # Check if port is in use and stop it
    $existing = Get-NetTCPConnection -LocalPort $script:ServerPort -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Status "Port in use - killing old process..." "Yellow"
        Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
        Start-Sleep 1
    }
    
    $env:ASPNETCORE_URLS = "http://localhost:$script:ServerPort"

    if ($script:PublishedExe -and (Test-Path $script:PublishedExe)) {
        Write-Status "Starting published binary..." "Cyan"
        $script:ServerProc = Start-Process -FilePath $script:PublishedExe -PassThru -WindowStyle Hidden
    } else {
        Write-Status "Starting with dotnet run..." "Yellow"
        $script:ServerProc = Start-Process -FilePath "dotnet" -ArgumentList "run","--project","$RepoRoot\FractalServe","-c","Release" -PassThru -WindowStyle Hidden
    }

    Write-Status "Waiting for server..." "Cyan"
    $ready = $false
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 400
        if (Get-NetTCPConnection -LocalPort $script:ServerPort -ErrorAction SilentlyContinue) {
            $ready = $true
            break
        }
    }

    if ($ready) {
        Write-Status "Server ONLINE on http://localhost:$script:ServerPort" "Green"
        Write-Host ""
        Write-Host "  SETUP COMPLETE! Fractal is now ready to use." -ForegroundColor Green
    } else {
        Write-Status "Server failed to start" "Red"
    }

    Pause-Enter
}

function Download-TinyLlama {
    Write-Banner
    Write-Host "  DOWNLOAD TINYLLAMA (test model)" -ForegroundColor Green
    Write-Host ""

    $targetDir  = Join-Path $script:RepoRoot "Models"
    $targetFile = Join-Path $targetDir "tinyllama-1.1b.safetensors"

    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }

    if (Test-Path $targetFile) {
        Write-Status "Already exists: $targetFile" "Green"
        $script:ModelPath = $targetFile
        $env:FRACTAL_MODEL = $targetFile
        Pause-Enter
        return
    }

    Write-Status "Downloading TinyLlama-1.1B (~2.2 GB)..." "Cyan"
    $url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors"

    try {
        Invoke-WebRequest -Uri $url -OutFile $targetFile -UseBasicParsing
        Write-Status "Download complete!" "Green"
        $script:ModelPath = $targetFile
        $env:FRACTAL_MODEL = $targetFile
        [Environment]::SetEnvironmentVariable("FRACTAL_MODEL", $targetFile, "User")
    } catch {
        Write-Status "Download failed: $_" "Red"
    }
    Pause-Enter
}

function Build-NativeAOT {
    Write-Banner
    Write-Host "  BUILD NativeAOT" -ForegroundColor Green
    Write-Host ""

    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
        Write-Status ".NET SDK not found" "Red"
        Pause-Enter
        return
    }

    Write-Status "Publishing FractalServe (this can take a minute)..." "Cyan"
    Push-Location $script:RepoRoot
    try {
        & dotnet publish FractalServe\FractalServe.csproj -c Release -r win-x64 /p:PublishAot=true --verbosity minimal
        if ($LASTEXITCODE -eq 0) {
            $script:PublishedExe = "$RepoRoot\FractalServe\bin\Release\net10.0\win-x64\publish\FractalServe.exe"
            Write-Status "Build succeeded" "Green"
        } else {
            Write-Status "Build failed" "Red"
        }
    } finally {
        Pop-Location
    }
    Pause-Enter
}

function Start-Server {
    Write-Banner
    Write-Host "  START SERVER" -ForegroundColor Green
    Write-Host ""

    if (-not $script:ModelPath -or -not (Test-Path $script:ModelPath)) {
        Write-Status "No model set. Use option 2 or 3 first." "Red"
        Pause-Enter
        return
    }

    $existing = Get-NetTCPConnection -LocalPort $script:ServerPort -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Status "Port in use - killing old process..." "Yellow"
        Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
        Start-Sleep 1
    }

    $env:FRACTAL_MODEL = $script:ModelPath
    $env:ASPNETCORE_URLS = "http://localhost:$script:ServerPort"

    if ($script:PublishedExe -and (Test-Path $script:PublishedExe)) {
        Write-Status "Starting published binary..." "Cyan"
        $script:ServerProc = Start-Process -FilePath $script:PublishedExe -PassThru -WindowStyle Hidden
    } else {
        Write-Status "Starting with dotnet run..." "Yellow"
        $script:ServerProc = Start-Process -FilePath "dotnet" -ArgumentList "run","--project","$RepoRoot\FractalServe","-c","Release" -PassThru -WindowStyle Hidden
    }

    Write-Status "Waiting for server..." "Cyan"
    $ready = $false
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 400
        if (Get-NetTCPConnection -LocalPort $script:ServerPort -ErrorAction SilentlyContinue) {
            $ready = $true
            break
        }
    }

    if ($ready) {
        Write-Status "Server ONLINE on http://localhost:$script:ServerPort" "Green"
    } else {
        Write-Status "Server failed to start" "Red"
    }
    Pause-Enter
}

function Launch-ChatTUI {
    Write-Banner
    Write-Host "  LAUNCH FRACTAL CHAT TUI" -ForegroundColor Green
    Write-Host ""

    if (-not (Get-NetTCPConnection -LocalPort $script:ServerPort -ErrorAction SilentlyContinue)) {
        Write-Status "Server is not running. Please start it first!" "Red"
        Pause-Enter
        return
    }

    Write-Status "Launching Fractal Chat (Spectre.Console)..." "Cyan"
    
    # Check if we need to build it
    $chatExe = "$script:RepoRoot\FractalChat\bin\Release\net10.0\win-x64\publish\FractalChat.exe"
    if (-not (Test-Path $chatExe)) {
        $chatExe = "$script:RepoRoot\FractalChat\bin\Release\net9.0\win-x64\publish\FractalChat.exe"
    }

    if (-not (Test-Path $chatExe)) {
        Write-Status "Building FractalChat TUI for the first time..." "Yellow"
        Push-Location $script:RepoRoot
        & dotnet publish FractalChat\FractalChat.csproj -c Release -r win-x64 /p:PublishAot=false --verbosity minimal
        Pop-Location
    }

    try {
        & dotnet run --project "$script:RepoRoot\FractalChat\FractalChat.csproj" -c Release
    } catch {
        Write-Host ""
        Write-Status "Chat exited or failed: $_" "Red"
    }
}

function Stop-Server {
    Write-Banner
    Write-Host "  STOP SERVER" -ForegroundColor Green
    Write-Host ""

    $conn = Get-NetTCPConnection -LocalPort $script:ServerPort -ErrorAction SilentlyContinue
    if ($conn) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Status "Server stopped" "Green"
    } else {
        Write-Status "No server running" "Yellow"
    }
    $script:ServerProc = $null
    Pause-Enter
}

# Main loop
while ($true) {
    Write-Banner

    $modelDisplay = if ($script:ModelPath) { Split-Path $script:ModelPath -Leaf } else { "(none)" }
    $serverDisplay = if (Get-NetTCPConnection -LocalPort $script:ServerPort -ErrorAction SilentlyContinue) { "ONLINE" } else { "stopped" }

    Write-Host "  Model  : $modelDisplay" -ForegroundColor Cyan
    Write-Host "  Server : $serverDisplay" -ForegroundColor $(if ($serverDisplay -eq "ONLINE") {"Green"} else {"Yellow"})
    Write-Host ""

    $options = @(
        "Easy Setup Wizard (Recommended)"
        "System check"
        "Set model path manually"
        "Download TinyLlama (safe test model)"
        "Build NativeAOT binary"
        "Start FractalServe"
        "Launch Interactive Chat TUI"
        "Stop server"
        "Exit"
    )

    $choice = Get-Choice -Options $options

    switch ($choice) {
        0 { Run-SetupWizard }
        1 { Show-SystemCheck }
        2 { Set-ModelPath }
        3 { Download-TinyLlama }
        4 { Build-NativeAOT }
        5 { Start-Server }
        6 { Launch-ChatTUI }
        7 { Stop-Server }
        8 { 
            Write-Host ""
            Write-Host "  Goodbye." -ForegroundColor Cyan
            exit 0
        }
    }
}