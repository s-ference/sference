"""Sference MCP server package."""

from importlib.metadata import PackageNotFoundError, version as _distribution_version


def _package_version(distribution_name: str) -> str:
    try:
        return _distribution_version(distribution_name)
    except PackageNotFoundError:
        return "0.0.0-dev"


__version__ = _package_version("sference-mcp")

__all__ = ["__version__"]
