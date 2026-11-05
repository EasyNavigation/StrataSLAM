#!/usr/bin/env python3
"""Fail when a dependency is unregistered, uses a forbidden license, or the registry is stale.

Usage: license_audit.py [--root PATH] [--registry THIRD_PARTY.md] [--ldd LIBRARY ...]
With --ldd, every shared library the given artifacts load must match an Ids pattern of the registry
(toolchain/OS runtime excepted) and every `transitive` row must be loaded by one of them.
Exit codes: 0 clean, 1 problems found.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ALLOWED = {"MIT", "BSD", "BSD-2", "BSD-2-Clause", "BSD-3", "BSD-3-Clause", "Apache-2.0"}
MPL_ONLY = {"MPL-2.0": {"Eigen"}}
# Permissive licenses outside the specification's list, accepted for any dependency (ADR-0051).
PERMISSIVE_EQUIVALENT = {"Zlib", "libpng-2.0", "IJG"}
# Permissive licenses accepted for libraries an allowed dependency's distribution package pulls in.
TRANSITIVE_ALLOWED = {"BSL-1.0", "X11"}
# Weak copyleft: accepted only for a library that a dependency's system package loads (`transitive`) or compiles in
# (`embedded`), linked dynamically and never redistributed by StrataSLAM (ADR-0051). Strong copyleft never is.
WEAK_COPYLEFT = re.compile(r"^(LGPL-(2\.1|3\.0)(-only|-or-later)?|MPL-2\.0)$")
# Optional tools that are never linked nor distributed (ADR-0036).
TOOL_ONLY = {"PSF-2.0": {"matplotlib"}}
# C/C++ toolchain and OS runtime (glibc, GCC runtime under its exception), the ROS 2 platform and
# the project's own libraries: not third-party code that the project distributes.
# libquadmath is not among them: it is LGPL-2.1+, without the GCC Runtime Library Exception.
RUNTIME_LIBS = ("linux-vdso", "ld-linux*", "libc", "libm", "libdl", "libpthread", "librt", "libstdc++",
                "libgcc_s", "libgomp", "libgfortran", "libstrataslam_*", "libsmall_gicp*")
STATUSES = ("active", "transitive", "embedded")
# Strong copyleft and non-free terms: forbidden for every row, transitive and embedded ones included.
FORBIDDEN_PATTERNS = re.compile(r"\b(A?GPL|non-?commercial|CC-BY-NC|proprietary)\b", re.IGNORECASE)

SKIP_DIRS = {".git", "build", "install", "log", "__pycache__", ".venv", "node_modules"}
# Dependencies that are part of the project itself or of the ROS 2 toolchain, not third party.
IGNORED_PREFIXES = ("strataslam_", "ament_", "rclcpp", "rclpy", "rcl_interfaces", "std_", "sensor_msgs", "nav_msgs",
                    "geometry_msgs", "tf2", "message_filters", "launch", "rosbag2", "rosidl",
                    "builtin_interfaces", "cmake", "python_cmake_module", "ros2bag", "ament")  # ROS 2 platform packages

# Compiler/OS runtime pieces, not distributed third-party code (OpenMP = libgomp, GPL + runtime exception).
IGNORED_CMAKE = {"openmp", "threads", "python3"}  # python3 = interpreter/headers from the OS (PSF license)

CMAKE_FIND = re.compile(r"find_package\(\s*([A-Za-z0-9_+\-]+)", re.IGNORECASE)
CMAKE_FETCH = re.compile(r"FetchContent_Declare\(\s*([A-Za-z0-9_+\-]+)", re.IGNORECASE)
XML_DEP = re.compile(r"<(?:depend|build_depend|exec_depend|test_depend|build_export_depend|buildtool_depend)"
                     r"[^>]*>\s*([^<\s]+)\s*<")
REQ_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


@dataclass
class Entry:
    name: str
    ids: set[str]
    license: str
    status: str


def parse_registry(path: Path) -> list[Entry]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"<!-- audit:table -->(.*?)<!-- /audit:table -->", text, re.DOTALL)
    if not m:
        raise SystemExit(f"{path}: missing <!-- audit:table --> markers")
    entries: list[Entry] = []
    rows = [r for r in m.group(1).strip().splitlines() if r.strip().startswith("|")]
    header = [c.strip().lower() for c in rows[0].strip("|").split("|")]
    for row in rows[2:]:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        rec = dict(zip(header, cells))
        entries.append(Entry(rec["name"], {i.strip() for i in rec["ids"].split(",") if i.strip()},
                             rec["license"], rec["status"].lower()))
    return entries


def license_terms(lic: str) -> list[str]:
    """The licenses of a row: an SPDX-like `A AND B` (every one applies), parenthesised notes removed."""
    return [t.strip() for t in re.split(r"\s+AND\s+", re.sub(r"\([^)]*\)", "", lic)) if t.strip()]


def term_problem(e: Entry, lic: str) -> str | None:
    """Why one license term of row `e` is not accepted, or None."""
    transitive = e.status in ("transitive", "embedded")
    if lic in ALLOWED or lic in PERMISSIVE_EQUIVALENT:
        return None
    if lic in MPL_ONLY and e.name in MPL_ONLY[lic]:
        return None
    if WEAK_COPYLEFT.match(lic):
        if transitive:
            return None
        if lic in MPL_ONLY:
            return f"{lic} is only allowed for {sorted(MPL_ONLY[lic])} or for transitive and embedded libraries (ADR-0051)"
        return f"forbidden license '{lic}' (weak copyleft is accepted only for transitive and embedded libraries, ADR-0051)"
    if lic in TOOL_ONLY:
        return None if e.name in TOOL_ONLY[lic] else f"{lic} is only allowed for {sorted(TOOL_ONLY[lic])}"
    if lic in TRANSITIVE_ALLOWED:
        return None if transitive else f"{lic} is only allowed for transitive libraries"
    return f"license '{lic}' is not on the allowed list"


def license_problems(e: Entry) -> list[str]:
    if FORBIDDEN_PATTERNS.search(e.license):
        return [f"{e.name}: forbidden license '{e.license}'"]
    return [f"{e.name}: {why}" for why in (term_problem(e, t) for t in license_terms(e.license)) if why]


def walk(root: Path):
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.is_file():
            yield p


def find_dependencies(root: Path) -> dict[str, set[str]]:
    """Map dependency id -> files that declare it."""
    found: dict[str, set[str]] = {}

    def add(dep: str, where: Path) -> None:
        found.setdefault(dep, set()).add(str(where.relative_to(root)))

    for p in walk(root):
        name = p.name
        try:
            if name == "CMakeLists.txt" or p.suffix == ".cmake":
                text = p.read_text(encoding="utf-8")
                for rx in (CMAKE_FIND, CMAKE_FETCH):
                    for dep in rx.findall(text):
                        if dep.lower() not in IGNORED_CMAKE and not dep.startswith(IGNORED_PREFIXES) and not dep.startswith("strataslam"):
                            add(dep, p)
            elif name == "package.xml":
                for dep in XML_DEP.findall(p.read_text(encoding="utf-8")):
                    if not dep.startswith(IGNORED_PREFIXES):
                        add(dep, p)
            elif name.startswith("requirements") and p.suffix == ".txt":
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip() and not line.lstrip().startswith("#"):
                        m = REQ_LINE.match(line)
                        if m:
                            add(m.group(1), p)
            elif name == "pyproject.toml":
                project = tomllib.loads(p.read_text(encoding="utf-8")).get("project", {})
                specs = list(project.get("dependencies", []))
                for extra in project.get("optional-dependencies", {}).values():
                    specs += extra
                for spec in specs:
                    m = REQ_LINE.match(spec)
                    if m:
                        add(m.group(1), p)
        except UnicodeDecodeError:
            continue
    return found


def linked_libraries(artifact: Path, unresolved: list[str] | None = None) -> set[str]:
    """Shared libraries `artifact` loads (ldd), as base names without the .so suffix. Libraries ldd cannot
    resolve ("not found") are appended to `unresolved`: what they would load cannot be checked."""
    out = subprocess.run(["ldd", str(artifact)], capture_output=True, text=True, check=True).stdout
    names = set()
    for line in out.splitlines():
        if line.strip():
            base = Path(line.split()[0]).name
            names.add(re.sub(r"\.so(\.[0-9.]*)?$", "", base))
            if "not found" in line and unresolved is not None:
                unresolved.append(f"{artifact.name} loads '{base}', which ldd cannot find: its dependencies were not checked "
                                  "(run with the environment the artifact runs in, e.g. ROS sourced)")
    return names


def ldd_problems(entries: list[Entry], libs: set[str]) -> list[str]:
    problems = []
    for lib in sorted(libs):
        if lib.startswith("/") or any(fnmatch.fnmatch(lib, p) for p in RUNTIME_LIBS):
            continue
        if not any(fnmatch.fnmatch(lib, i) for e in entries for i in e.ids):
            problems.append(f"unregistered linked library '{lib}'")
    for e in entries:
        if e.status == "transitive" and not any(fnmatch.fnmatch(lib, i) for lib in libs for i in e.ids):
            problems.append(f"{e.name}: marked transitive but not loaded by the checked artifacts")
    return problems


def audit(root: Path, registry: Path, artifacts: list[Path] | None = None) -> list[str]:
    entries = parse_registry(registry)
    problems: list[str] = []
    for e in entries:
        problems += license_problems(e)
    registered = {i.lower(): e for e in entries for i in e.ids}
    found = find_dependencies(root)
    for dep, files in sorted(found.items()):
        if dep.lower() not in registered:
            problems.append(f"unregistered dependency '{dep}' (declared in {', '.join(sorted(files))})")
    used = {d.lower() for d in found}
    for e in entries:
        if e.status == "active" and not any(i.lower() in used for i in e.ids):
            problems.append(f"{e.name}: marked active but not used by any build file")
        if e.status not in STATUSES:
            problems.append(f"{e.name}: unknown status '{e.status}' ({', '.join(STATUSES)})")
    if artifacts:
        libs: set[str] = set()
        unresolved: list[str] = []
        for a in artifacts:
            libs |= linked_libraries(a, unresolved)
        problems += unresolved
        problems += ldd_problems(entries, libs)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--ldd", type=Path, nargs="*", default=[], help="built libraries/executables to check with ldd")
    args = ap.parse_args()
    registry = args.registry or args.root / "THIRD_PARTY.md"
    problems = audit(args.root, registry, args.ldd)
    for p in problems:
        print(f"license-audit: {p}", file=sys.stderr)
    if problems:
        return 1
    print("license-audit: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
