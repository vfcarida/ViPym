"""Environment sanitizer and secure credential scrubber."""

import os

SENSITIVE_ENV_PREFIXES: set[str] = {
    "AWS_",
    "HF_",
    "HUGGING_FACE",
    "OPENAI_",
    "GITHUB_",
    "SSH_",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "KEY",
}


def sanitize_execution_environment() -> dict[str, str]:
    """Return a sanitized, minimal environment dict with all sensitive credentials stripped."""
    clean_env: dict[str, str] = {
        "PYTHONPATH": "",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }

    safe_keys = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "SystemRoot",
        "WINDIR",
        "windir",
        "COMSPEC",
        "ComSpec",
        "TEMP",
        "TMP",
        "TMPDIR",
        "SYSTEMDRIVE",
        "SystemDrive",
        "PROGRAMDATA",
        "ProgramData",
        "PROGRAMFILES",
        "ProgramFiles",
        "PROGRAMFILES(X86)",
        "USERPROFILE",
        "HOME",
        "HOMEPATH",
        "HOMEDRIVE",
    }

    for key, val in os.environ.items():
        if any(key.upper().startswith(p) for p in SENSITIVE_ENV_PREFIXES):
            continue
        if key in safe_keys:
            clean_env[key] = val

    if "PATH" not in clean_env:
        clean_env["PATH"] = "/usr/bin:/bin"

    return clean_env
