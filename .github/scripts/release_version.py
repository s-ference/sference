"""Release version guards for the SDK and CLI packages.

Usage:
  release_version.py <package-dir> <tag>
      Fail unless <tag> is `<prefix>v<version>` for the package's version, so a
      mistyped tag cannot publish the wrong version.
  release_version.py check-bumps <base-ref>
      Fail when a package's source changed since <base-ref> but its version did
      not, because an unbumped release silently never reaches PyPI.
  release_version.py notes <package-dir> <tag>
      Print GitHub release notes for <tag>: the merged PRs that changed the
      shipped package (tests excluded) since the package's previous release.
"""

from __future__ import annotations

import os
import re
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
        print(
            f"::error::tag {tag!r} does not match {package_dir} version; expected {expected!r}"
        )
        return 1
    return 0


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def check_bumps(base_ref: str) -> int:
    failed = False
    for package_dir, (_, source_dir) in PACKAGES.items():
        if not _git(
            "diff", "--name-only", f"{base_ref}...HEAD", "--", source_dir
        ).strip():
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


def _previous_tag(package_dir: str, tag: str) -> str | None:
    """The package's last release before ``tag``.

    Releases before per-package tags were shared ``v<version>`` tags, so fall
    back to the newest of those reachable from ``tag``.
    """
    prefix, _ = PACKAGES[package_dir]
    older = [
        t
        for t in _git(
            "tag", "--list", f"{prefix}*", "--sort=-v:refname", "--merged", tag
        ).split()
        if t != tag
    ]
    if older:
        return older[0]
    try:
        return (
            _git(
                "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", f"{tag}^"
            ).strip()
            or None
        )
    except subprocess.CalledProcessError:
        return None


# A merge commit ("Merge pull request #N from ...", PR title in the body) or a
# squash merge ("<title> (#N)").
_MERGE = re.compile(r"^Merge pull request #(\d+) from \S+$")
_SQUASH = re.compile(r"^(.*) \(#(\d+)\)$")


def release_notes(package_dir: str, tag: str, repo: str) -> str:
    previous = _previous_tag(package_dir, tag)
    span = f"{previous}..{tag}" if previous else tag
    # First-parent history of main: one entry per merged PR. Tests don't ship,
    # so a test-only PR is left out.
    log = _git(
        "log",
        "--first-parent",
        "--format=%s%x1f%b%x1e",
        span,
        "--",
        package_dir,
        f":(exclude){package_dir}/tests",
    )
    changes: list[str] = []
    for entry in filter(None, (e.strip() for e in log.split("\x1e"))):
        subject, _, body = entry.partition("\x1f")
        if m := _MERGE.match(subject):
            title = body.strip().splitlines()[0] if body.strip() else subject
            changes.append(f"- {title} (#{m.group(1)})")
        elif m := _SQUASH.match(subject):
            changes.append(f"- {m.group(1)} (#{m.group(2)})")
        else:
            changes.append(f"- {subject}")
    name = "sference-sdk" if package_dir == "sdk-python" else "sference-cli"
    lines = [
        f"`pip install -U {name}`",
        "",
        "## Changes",
        *(changes or ["- No changes to the package source."]),
    ]
    if previous:
        lines += [
            "",
            f"**Full diff:** https://github.com/{repo}/compare/{previous}...{tag}",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "check-bumps":
        return check_bumps(argv[2])
    if len(argv) == 4 and argv[1] == "notes" and argv[2] in PACKAGES:
        # Set by Actions; the default keeps local dry runs working.
        repo = os.environ.get("GITHUB_REPOSITORY", "s-ference/sference")
        print(release_notes(argv[2], argv[3], repo), end="")
        return 0
    if len(argv) == 3 and argv[1] in PACKAGES:
        return check_tag(argv[1], argv[2])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
