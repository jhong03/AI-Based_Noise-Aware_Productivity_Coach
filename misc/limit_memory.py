# limit_memory.py
# Hard-cap the entire Python process (and all child processes) memory on Windows.
# Safe no-op on non-Windows.

import os
import sys

def _is_windows() -> bool:
    return os.name == "nt"

if _is_windows():
    import ctypes
    from ctypes import wintypes

    # Flags
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_JOB_MEMORY     = 0x00000200

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount",   ctypes.c_ulonglong),
            ("WriteOperationCount",  ctypes.c_ulonglong),
            ("OtherOperationCount",  ctypes.c_ulonglong),
            ("ReadTransferCount",    ctypes.c_ulonglong),
            ("WriteTransferCount",   ctypes.c_ulonglong),
            ("OtherTransferCount",   ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit",     ctypes.c_longlong),
            ("LimitFlags",              ctypes.c_uint),
            ("MinimumWorkingSetSize",   ctypes.c_size_t),
            ("MaximumWorkingSetSize",   ctypes.c_size_t),
            ("ActiveProcessLimit",      ctypes.c_uint),
            ("Affinity",                ctypes.c_size_t),
            ("PriorityClass",           ctypes.c_uint),
            ("SchedulingClass",         ctypes.c_uint),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo",                IO_COUNTERS),
            ("ProcessMemoryLimit",    ctypes.c_size_t),
            ("JobMemoryLimit",        ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed",     ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    CreateJobObjectW            = kernel32.CreateJobObjectW
    CreateJobObjectW.argtypes   = [wintypes.LPVOID, wintypes.LPCWSTR]
    CreateJobObjectW.restype    = wintypes.HANDLE

    SetInformationJobObject     = kernel32.SetInformationJobObject
    SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    SetInformationJobObject.restype  = wintypes.BOOL

    AssignProcessToJobObject    = kernel32.AssignProcessToJobObject
    AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    AssignProcessToJobObject.restype  = wintypes.BOOL

    GetCurrentProcess           = kernel32.GetCurrentProcess
    GetCurrentProcess.argtypes  = []
    GetCurrentProcess.restype   = wintypes.HANDLE

    JobObjectExtendedLimitInformation = 9  # per Windows SDK

def apply_memory_cap_mb(max_mb: int) -> bool:
    """
    Apply a HARD memory cap to the whole process tree (Windows only).
    Returns True if applied, False otherwise.
    """
    if not _is_windows():
        # No-op on non-Windows
        return False

    try:
        max_bytes = int(max_mb) * 1024 * 1024

        hJob = CreateJobObjectW(None, None)
        if not hJob:
            raise OSError("CreateJobObjectW failed")

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        # Set flags to enforce both per-process and whole-job memory caps
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
        info.ProcessMemoryLimit = max_bytes
        info.JobMemoryLimit     = max_bytes

        size = ctypes.sizeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
        if not SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, ctypes.byref(info), size):
            raise OSError("SetInformationJobObject failed")

        hProcess = GetCurrentProcess()
        if not AssignProcessToJobObject(hJob, hProcess):
            raise OSError("AssignProcessToJobObject failed")

        print(f"🔒 OS hard memory cap applied: {max_mb} MB (job + per-process)")
        return True
    except Exception as e:
        print(f"⚠️ Failed to apply OS memory cap: {e}")
        return False
