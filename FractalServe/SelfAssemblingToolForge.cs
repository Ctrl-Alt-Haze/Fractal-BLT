using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace FractalServe;

/// <summary>
/// Phase 7: Self-Assembling Tool Forge (Native Tool JIT)
/// Allows the agent to write, compile, and execute hardware-native C# tools at runtime.
/// Triggers a background `dotnet build` to create a native library, then injects it instantly.
/// </summary>
public class SelfAssemblingToolForge
{
    private readonly string _toolsDirectory;

    public SelfAssemblingToolForge()
    {
        _toolsDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ForgeTools");
        if (!Directory.Exists(_toolsDirectory))
        {
            Directory.CreateDirectory(_toolsDirectory);
        }
    }

    /// <summary>
    /// Writes the raw C# code to a temporary csproj, builds it, and returns the path to the native DLL.
    /// This runs asynchronously to prevent blocking the GPU hot path.
    /// </summary>
    public async Task<string> ForgeNativeToolAsync(string toolName, string csharpCode)
    {
        Console.WriteLine($"[Tool Forge] Forging native tool: {toolName}...");
        string toolDir = Path.Combine(_toolsDirectory, toolName);
        Directory.CreateDirectory(toolDir);

        string csPath = Path.Combine(toolDir, $"{toolName}.cs");
        await File.WriteAllTextAsync(csPath, csharpCode);

        // We compile it as a standard .NET library, which can export [UnmanagedCallersOnly] methods
        // NativeAOT can be triggered by adding <PublishAot>true</PublishAot>, but for instant JIT during a turn,
        // standard shared libraries with NativeLibrary interop often compile faster. 
        // For strict "NativeAOT" compliance as requested:
        string csprojContent = $@"
<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <PublishAot>true</PublishAot>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  </PropertyGroup>
</Project>";
        
        string csprojPath = Path.Combine(toolDir, $"{toolName}.csproj");
        await File.WriteAllTextAsync(csprojPath, csprojContent);

        // Trigger dotnet build in the background
        var processInfo = new ProcessStartInfo
        {
            FileName = "dotnet",
            Arguments = $"publish -c Release -r win-x64",
            WorkingDirectory = toolDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = Process.Start(processInfo);
        await process!.WaitForExitAsync();

        if (process.ExitCode != 0)
        {
            string errors = await process.StandardError.ReadToEndAsync();
            throw new Exception($"[Tool Forge] Compilation failed for {toolName}:\n{errors}");
        }

        string nativeDllPath = Path.Combine(toolDir, "bin", "Release", "net10.0", "win-x64", "publish", $"{toolName}.dll");
        Console.WriteLine($"[Tool Forge] Successfully compiled {toolName} to native library at {nativeDllPath}");
        
        return nativeDllPath;
    }

    /// <summary>
    /// Injects the freshly compiled native tool into the engine's memory space and executes a specific unmanaged entry point.
    /// </summary>
    public unsafe void ExecuteNativeTool(string nativeDllPath, string entryPointName)
    {
        Console.WriteLine($"[Tool Forge] Injecting and executing {entryPointName} from {nativeDllPath}...");

        IntPtr libraryHandle = NativeLibrary.Load(nativeDllPath);
        if (libraryHandle == IntPtr.Zero)
        {
            throw new Exception($"[Tool Forge] Failed to load native library: {nativeDllPath}");
        }

        try
        {
            IntPtr exportAddress = NativeLibrary.GetExport(libraryHandle, entryPointName);
            if (exportAddress == IntPtr.Zero)
            {
                throw new Exception($"[Tool Forge] Failed to locate unmanaged entry point '{entryPointName}'");
            }

            // Cast the raw memory address to a C# function pointer and invoke it with zero overhead
            delegate* unmanaged[Cdecl]<void> toolMethod = (delegate* unmanaged[Cdecl]<void>)exportAddress;
            toolMethod();
            
            Console.WriteLine($"[Tool Forge] Tool '{entryPointName}' executed successfully.");
        }
        finally
        {
            // Free the library so the file isn't locked if we want to recompile/overwrite it later
            NativeLibrary.Free(libraryHandle);
        }
    }
}
