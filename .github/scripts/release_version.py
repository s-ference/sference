"""Release version guards for the SDK and CLI packages.

Usage:
  release_version.py <package-dir> <tag>
      Fail unless <tag> is `<prefix>v<version>` for the package's version, so a
      mistyped tag cannot publish the wrong version.
  release_version.py check-bumps <base-ref>
      Fail when a package's source changed since <base-ref> but its version did
      not, because an unbumped release silently never reaches PyPI.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib

# package dir -> (tag prefix, source dir whose changes require a version bump)
PACKAGES = {
    "sdk-python": ("sdk-v", "sdk-python/sference_sdk/"),
    "cli": ("cli-v", "cli/sference_cli/"),
}


def _version(toml_text: str) -> str:
    return tomllib.loads(toml_text)["project"]["version"]


def check_tag(package_dir: str, tag: str) -> int:
    prefix, _ = PACKAGES[package_dir]
    with open(f"{package_dir}/pyproject.toml", encoding="utf-8") as f:
        version = _version(f.read())
    expected = f"{prefix}{version}"
    if tag != expected:
        print(f"::error::tag {tag!r} does not match {package_dir} version; expected {expected!r}")
        return 1
    return 0


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def check_bumps(base_ref: str) -> int:
    failed = False
    for package_dir, (_, source_dir) in PACKAGES.items():
        if not _git("diff", "--name-only", f"{base_ref}...HEAD", "--", source_dir).strip():
            continue
        pyproject = f"{package_dir}/pyproject.toml"
        before = _version(_git("show", f"{base_ref}:{pyproject}"))
        with open(pyproject, encoding="utf-8") as f:
            after = _version(f.read())
        if before == after:
            print(
                f"::error file={pyproject}::{source_dir} changed but the version is still "
                f"{before}. Bump it, or add the 'no-release' label if this change "
                "does not need publishing."
            )
            failed = True
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "check-bumps":
        return check_bumps(argv[2])
    if len(argv) == 3 and argv[1] in PACKAGES:
        return check_tag(argv[1], argv[2])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
