"""Environment sanitizer and secure credential scrubber."""

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
    clean_env = {
        "PYTHONPATH": "",
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    return clean_env
