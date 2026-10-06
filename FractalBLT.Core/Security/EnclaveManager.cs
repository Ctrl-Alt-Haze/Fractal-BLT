using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace FractalBLT.Core.Security;

/// <summary>
/// Phase 14: Opure Zero-Trust Agent Enclave (OZTAE)
/// Enforces strict containment of dynamically generated NativeAOT code via Windows Job Objects.
/// </summary>
public unsafe class EnclaveManager : IDisposable
{
    private IntPtr _jobHandle;
    private readonly string _ledgerDbPath;
    private readonly string _sandboxDir;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr hJob, int JobObjectInformationClass, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION lpJobObjectInformation, uint cbJobObjectInformationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("e_sqlite3", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    private static extern int sqlite3_open_v2(byte* filename, out IntPtr ppDb, int flags, byte* zVfs);

    [DllImport("e_sqlite3", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    private static extern int sqlite3_exec(IntPtr db, byte* sql, IntPtr callback, IntPtr arg, out IntPtr errmsg);

    [DllImport("e_sqlite3", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    private static extern int sqlite3_close(IntPtr db);

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
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
    private struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public nuint ProcessMemoryLimit;
        public nuint JobMemoryLimit;
        public nuint PeakProcessMemoryUsed;
        public nuint PeakJobMemoryUsed;
    }

    private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    private const uint JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100;
    private const uint JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008;
    
    private const int SQLITE_OPEN_READWRITE = 0x00000002;
    private const int SQLITE_OPEN_CREATE = 0x00000004;

    private IntPtr _dbHandle;

    public EnclaveManager(string sandboxDirectory)
    {
        _sandboxDir = sandboxDirectory;
        if (!Directory.Exists(_sandboxDir)) Directory.CreateDirectory(_sandboxDir);

        _ledgerDbPath = Path.Combine(_sandboxDir, "oztae_ledger.db");
        InitializeLedger();

        // Establish the secure Job Object Enclave
        _jobHandle = CreateJobObject(IntPtr.Zero, "Opure_OZTAE_" + Guid.NewGuid().ToString("N"));
        
        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        // Limit to 1 active process (no fork bombs), strict memory limits, and auto-kill
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.BasicLimitInformation.ActiveProcessLimit = 1;
        info.ProcessMemoryLimit = 512 * 1024 * 1024; // 512 MB max for dynamically generated tool executions
        
        SetInformationJobObject(_jobHandle, 9, ref info, (uint)Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)));
    }

    private void InitializeLedger()
    {
        byte[] dbPathBytes = Encoding.UTF8.GetBytes(_ledgerDbPath + "\0");
        fixed (byte* pDbPath = dbPathBytes)
        {
            sqlite3_open_v2(pDbPath, out _dbHandle, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, null);
            
            string initSql = "CREATE TABLE IF NOT EXISTS AuditLog (Id INTEGER PRIMARY KEY AUTOINCREMENT, Timestamp TEXT, Action TEXT, TargetHash TEXT);";
            byte[] sqlBytes = Encoding.UTF8.GetBytes(initSql + "\0");
            fixed (byte* pSql = sqlBytes)
            {
                sqlite3_exec(_dbHandle, pSql, IntPtr.Zero, IntPtr.Zero, out _);
            }
        }
    }

    public void ContainProcess(Process proc)
    {
        if (proc != null && !proc.HasExited)
        {
            AssignProcessToJobObject(_jobHandle, proc.Handle);
        }
    }

    public void AuditToolExecution(string toolPath)
    {
        // 1. Verify Air-Gapped Path Boundary
        string fullPath = Path.GetFullPath(toolPath);
        string fullSandbox = Path.GetFullPath(_sandboxDir);
        if (!fullPath.StartsWith(fullSandbox, StringComparison.OrdinalIgnoreCase))
        {
            throw new AccessViolationException($"[OZTAE] SECURITY BREACH DETECTED: Execution attempted outside enclave boundary -> {fullPath}");
        }

        // 2. Cryptographic Hash of Tool Executable
        string hashStr;
        using (var sha256 = SHA256.Create())
        using (var stream = File.OpenRead(fullPath))
        {
            byte[] hash = sha256.ComputeHash(stream);
            hashStr = BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
        }

        // 3. Unmanaged SQLite Audit Logging
        string insertSql = $"INSERT INTO AuditLog (Timestamp, Action, TargetHash) VALUES ('{DateTime.UtcNow:O}', 'EXECUTE', '{hashStr}');";
        byte[] sqlBytes = Encoding.UTF8.GetBytes(insertSql + "\0");
        fixed (byte* pSql = sqlBytes)
        {
            sqlite3_exec(_dbHandle, pSql, IntPtr.Zero, IntPtr.Zero, out _);
        }
    }

    public void Dispose()
    {
        if (_dbHandle != IntPtr.Zero)
        {
            sqlite3_close(_dbHandle);
            _dbHandle = IntPtr.Zero;
        }
    }
}
