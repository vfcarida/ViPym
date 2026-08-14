# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in ViPym, please report it confidentially via GitHub Security Advisories or by emailing security@vipym.org.

## Sandboxing Untrusted Code

ViPym is designed to execute generated model code in isolated sandbox environments. Never disable gVisor / Docker isolation (`isolate_with_gvisor: false`) when running untrusted model checkpoints or public coding benchmarks in production environments.
