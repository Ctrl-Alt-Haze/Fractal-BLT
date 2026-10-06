using System;
using System.Runtime.InteropServices;

namespace FractalBridge;

public static partial class CuFileNative
{
    private const string CuFileLib = "cufile"; // libcufile.so or cufile.dll

    public enum CUfileError_t
    {
        CU_FILE_SUCCESS = 0
    }

    public enum CUfileHandleType : int
    {
        CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1,
        CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct CUfileDescr_t
    {
        [FieldOffset(0)]
        public CUfileHandleType type;

        [FieldOffset(8)]
        public int fd; // Linux

        [FieldOffset(8)]
        public IntPtr handle; // Windows
    }

    [LibraryImport(CuFileLib)]
    public static partial CUfileError_t cuFileDriverOpen();

    [LibraryImport(CuFileLib)]
    public static partial CUfileError_t cuFileDriverClose();

    [LibraryImport(CuFileLib)]
    public static partial CUfileError_t cuFileHandleRegister(out IntPtr cf_handle, ref CUfileDescr_t descr);

    [LibraryImport(CuFileLib)]
    public static partial void cuFileHandleDeregister(IntPtr cf_handle);

    [LibraryImport(CuFileLib)]
    public static partial IntPtr cuFileRead(IntPtr cf_handle, IntPtr devPtr_base, nuint size, long file_offset, long devPtr_offset);

    public static void Check(CUfileError_t result)
    {
        if (result != CUfileError_t.CU_FILE_SUCCESS)
            throw new Exception($"cuFile error: {result}");
    }
}
