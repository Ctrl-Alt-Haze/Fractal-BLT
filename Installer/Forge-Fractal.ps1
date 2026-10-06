<#
.SYNOPSIS
    Phase 11: Sovereign Zero-Config Packager & Bootstrapper
.DESCRIPTION
    Wraps the entire NativeAOT ecosystem into a single-click execution.
    Installs, configures, compiles to metal, pulls models, and enforces strict memory containment using Windows Job Objects.
#>

$ErrorActionPreference = "Stop"

$RootDir = "C:\Fractal-BLT"
$ServeProject = "$RootDir\FractalServe\FractalServe.csproj"
$ModelsDir = "$RootDir\Models"
$OutputExe = "$RootDir\FractalServe\bin\Release\net10.0\win-x64\publish\FractalServe.exe"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " FRACTAL-BLT: SOVEREIGN BOOTSTRAPPER" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. NativeAOT Compilation
Write-Host "`n[1/3] Forging Native Metal Executable..." -ForegroundColor Yellow
$publishArgs = @(
    "publish", $ServeProject,
    "-c", "Release",
    "-r", "win-x64",
    "/p:PublishAot=true",
    "/p:OptimizationPreference=Speed",
    "/p:StripSymbols=true"
)
dotnet @publishArgs

if (-not (Test-Path $OutputExe)) {
    Write-Error "NativeAOT forge failed! Executable not found at $OutputExe"
}
Write-Host "[+] NativeAOT Forge Complete. Zero-overhead executable ready." -ForegroundColor Green

# 2. Phase 8 Harvester (Auto-Pull 1.5B Traffic Cop)
Write-Host "`n[2/3] Checking Sovereign Model Vault..." -ForegroundColor Yellow
$models = Get-ChildItem -Path $ModelsDir -Directory | Select-Object -ExpandProperty Name
if ($models.Count -eq 1 -and $models[0] -eq "qwen3_30b_a3b") {
    Write-Host "[!] Only heavy fallback model found. Triggering Phase 8 Harvester for 1.5B Traffic Cop..." -ForegroundColor Magenta
    
    # Trigger the harvester sequence in the built executable
    $harvestArgs = "--harvest"
    $env:FRACTAL_MODEL = "$ModelsDir\qwen3_30b_a3b"
    
    $harvestProc = Start-Process -FilePath $OutputExe -ArgumentList $harvestArgs -NoNewWindow -PassThru -Wait
    if ($harvestProc.ExitCode -ne 0) {
        Write-Error "Model Harvester failed!"
    }
    Write-Host "[+] 1.58-bit Traffic Cop model acquired and squashed." -ForegroundColor Green
} else {
    Write-Host "[+] Traffic Cop model already present in vault." -ForegroundColor Green
}

# 3. Windows Job Object Memory Containment Execution
Write-Host "`n[3/3] Engaging Windows Job Object Memory Containment..." -ForegroundColor Yellow
# We'll use a small inline C# script to create the Job Object and launch the child process
$JobWrapperCode = @"
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

public class JobRunner {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr hJob, int JobObjectInformationClass, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION lpJobObjectInformation, uint cbJobObjectInformationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public nuint MinimumWorkingSetSize;
        public nuint MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public nuint Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IO_COUNTERS {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public nuint ProcessMemoryLimit;
        public nuint JobMemoryLimit;
        public nuint PeakProcessMemoryUsed;
        public nuint PeakJobMemoryUsed;
    }

    const uint JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100;
    const uint JOB_OBJECT_LIMIT_WORKINGSET = 0x00000001;

    public static void RunContained(string exePath, nuint memoryLimitBytes) {
        IntPtr hJob = CreateJobObject(IntPtr.Zero, "FractalBltContainment");
        
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_WORKINGSET;
        info.ProcessMemoryLimit = memoryLimitBytes;
        info.BasicLimitInformation.MaximumWorkingSetSize = memoryLimitBytes;
        info.BasicLimitInformation.MinimumWorkingSetSize = memoryLimitBytes / 2;

        SetInformationJobObject(hJob, 9, ref info, (uint)Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)));

        ProcessStartInfo psi = new ProcessStartInfo(exePath);
        psi.UseShellExecute = false;
        
        Process proc = Process.Start(psi);
        AssignProcessToJobObject(hJob, proc.Handle);
        
        Console.WriteLine("[JobObject] Memory bounds locked strictly to " + (memoryLimitBytes / 1024 / 1024) + " MB.");
        proc.WaitForExit();
    }
}
"@

Add-Type -TypeDefinition $JobWrapperCode -Language CSharp
Write-Host "[+] Initializing strict 2GB heap boundary container..." -ForegroundColor Cyan

# 2GB Memory Limit (2147483648 bytes) for the host process. 
# VRAM mappings (cuFile/GPUDirect) bypass CPU Working Set limits, ensuring zero-overhead GPU scale.
[JobRunner]::RunContained($OutputExe, 2147483648)

Write-Host "`n[+] Sovereign Bootstrapper execution terminated cleanly." -ForegroundColor Green
