# ADR-004: Open-Source Licensing Compatibility Assessment (Apache-2.0 & MIT)

## Status
Accepted / Documented

## Context
The repository was initialized with an `Apache-2.0` `LICENSE` file and `pyproject.toml` metadata specifying `Apache-2.0`. The prompt context mentions `PRIVATE_REPOSITORY: No. MIT license`.

## Analysis & Assessment
- Both **Apache-2.0** and **MIT** are permissive, OSI-approved open-source licenses granting broad rights for commercial use, modification, distribution, and sublicensing.
- **Apache-2.0** includes explicit patent grant and termination provisions, which are standard for modern enterprise-grade ML frameworks (e.g. PyTorch, vLLM, Apache TVM).
- **MIT** offers maximum simplicity with minimal restrictions.
- Inbound dependencies (PyTorch, Transformers, vLLM, Typer, Pydantic) are permissively licensed (BSD, Apache-2.0, MIT) and fully compatible with both licenses.

## Decision
Retain Apache-2.0 in `LICENSE` and `pyproject.toml` while ensuring all first-party code remains clean, permissive, and dual-compatible with MIT terms without proprietary patent encumbrances.
