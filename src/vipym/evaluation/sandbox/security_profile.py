"""Security profiles, resource limits, and seccomp configurations for sandboxing."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SandboxSecurityConfig:
    """Security configuration enforcing strict isolation."""
    timeout_seconds: int = 15
    memory_limit_mb: int = 2048
    cpu_quota: float = 2.0
    pids_limit: int = 100
    read_only_rootfs: bool = True
    network_disabled: bool = True
    drop_capabilities: List[str] = None
    use_gvisor_runsc: bool = True

    def __post_init__(self):
        if self.drop_capabilities is None:
            object.__setattr__(self, "drop_capabilities", ["ALL"])


SECCOMP_STRICT_JSON = """{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_X86",
    "SCMP_ARCH_AARCH64"
  ],
  "syscalls": [
    {
      "names": [
        "read", "write", "open", "openat", "close", "stat", "fstat", "lstat",
        "poll", "lseek", "mmap", "mprotect", "munmap", "brk", "rt_sigaction",
        "rt_sigprocmask", "rt_sigreturn", "ioctl", "access", "pipe", "select",
        "sched_yield", "mremap", "dup", "dup2", "getpid", "exit", "exit_group",
        "wait4", "fcntl", "gettimeofday", "getrlimit", "getrusage", "nanosleep",
        "clock_gettime", "clock_getres", "getuid", "getgid", "geteuid", "getegid"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}"""
