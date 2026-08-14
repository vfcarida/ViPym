# ADR-004: Open-Source Licensing Standardization (MIT License)

## Status
Accepted / Enacted

## Context
The project was harmonized to standardize definitively on the **MIT License** across `LICENSE`, `pyproject.toml`, `src/vipym/__version__.py`, and `README.md`.

## Analysis & Assessment
- Both **Apache-2.0** and **MIT** are permissive, OSI-approved open-source licenses granting broad rights for commercial use, modification, distribution, and sublicensing.
- **MIT** offers maximum simplicity, lightweight integration, and zero friction for adoption across academic research and enterprise engineering.
- Inbound dependencies (PyTorch, Transformers, vLLM, Typer, Pydantic) are permissively licensed (BSD, Apache-2.0, MIT) and 100% compatible with MIT distribution.

## Decision
Adopt and standardize the **MIT License** across all first-party repository artifacts, package metadata, and documentation.
