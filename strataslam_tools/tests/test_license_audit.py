"""Tests for scripts/license_audit.py using throwaway repositories."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("license_audit", REPO / "scripts" / "license_audit.py")
assert spec and spec.loader
audit_mod = importlib.util.module_from_spec(spec)
sys.modules["license_audit"] = audit_mod
spec.loader.exec_module(audit_mod)

HEADER = "<!-- audit:table -->\n| Name | Ids | Version | License | Use | Source | Status |\n| --- | --- | --- | --- | --- | --- | --- |\n"
FOOTER = "<!-- /audit:table -->\n"


def make_repo(tmp_path: Path, rows: list[str], cmake: str = "", pyproject: str = "") -> Path:
    (tmp_path / "THIRD_PARTY.md").write_text(HEADER + "\n".join(rows) + "\n" + FOOTER)
    if cmake:
        (tmp_path / "CMakeLists.txt").write_text(cmake)
    if pyproject:
        (tmp_path / "pyproject.toml").write_text(pyproject)
    return tmp_path


def run(root: Path) -> list[str]:
    return audit_mod.audit(root, root / "THIRD_PARTY.md")


def test_clean_repo(tmp_path: Path) -> None:
    root = make_repo(tmp_path, ["| Eigen | Eigen3 | 3.4 | MPL-2.0 | la | apt | active |"],
                     "find_package(Eigen3 REQUIRED)")
    assert run(root) == []


def test_unregistered_cmake_dependency_fails(tmp_path: Path) -> None:
    root = make_repo(tmp_path, [], "find_package(Boost REQUIRED)\nFetchContent_Declare(foo)")
    problems = run(root)
    assert any("unregistered dependency 'Boost'" in p for p in problems)
    assert any("unregistered dependency 'foo'" in p for p in problems)


def test_forbidden_licenses_fail(tmp_path: Path) -> None:
    for lic in ["GPL-3.0", "LGPL-2.1", "AGPL-3.0", "CC-BY-NC-4.0", "non-commercial"]:
        root = make_repo(tmp_path, [f"| X | x | 1 | {lic} | u | s | active |"])
        assert any("forbidden" in p for p in run(root)), lic


def test_mpl_only_for_eigen(tmp_path: Path) -> None:
    root = make_repo(tmp_path, ["| Other | o | 1 | MPL-2.0 | u | s | active |"])
    assert any("only allowed for" in p for p in run(root))


def test_unknown_license_fails(tmp_path: Path) -> None:
    root = make_repo(tmp_path, ["| X | x | 1 | WTFPL | u | s | active |"])
    assert any("not on the allowed list" in p for p in run(root))


def test_python_dependencies_detected(tmp_path: Path) -> None:
    py = '[project]\nname="t"\ndependencies = ["numpy>=1"]\n[project.optional-dependencies]\ndev = ["leftpad"]\n'
    root = make_repo(tmp_path, ["| numpy | numpy | 1 | BSD-3-Clause | u | pip | active |"], pyproject=py)
    problems = run(root)
    assert problems == ["unregistered dependency 'leftpad' (declared in pyproject.toml)"]


def test_stale_registry_entries_flagged(tmp_path: Path) -> None:
    root = make_repo(tmp_path, ["| A | a | 1 | MIT | u | s | active |", "| B | b | 1 | MIT | u | s | planned |"],
                     "find_package(b)")
    problems = run(root)
    assert any("A: marked active but not used" in p for p in problems)
    assert any("B: unknown status 'planned'" in p for p in problems)


def test_transitive_licenses_only_for_transitive_rows(tmp_path: Path) -> None:
    def repo(name: str, rows: list[str], cmake: str = "", pyproject: str = "") -> Path:
        (tmp_path / name).mkdir()
        return make_repo(tmp_path / name, rows, cmake, pyproject)

    assert run(repo("a", ["| Boost | libboost_* | 1 | BSL-1.0 | u | s | transitive |"])) == []
    problems = run(repo("b", ["| Boost | boost | 1 | BSL-1.0 | u | s | active |"], "find_package(boost)"))
    assert any("only allowed for transitive" in p for p in problems)
    py = '[project]\nname="t"\ndependencies = ["{}"]\n'
    assert run(repo("c", ["| matplotlib | matplotlib | 1 | PSF-2.0 | u | s | active |"], pyproject=py.format("matplotlib"))) == []
    problems = run(repo("d", ["| other | other | 1 | PSF-2.0 | u | s | active |"], pyproject=py.format("other")))
    assert any("only allowed for ['matplotlib']" in p for p in problems)


def test_linked_libraries_are_checked_against_the_registry() -> None:
    entries = audit_mod.parse_registry(REPO / "THIRD_PARTY.md")
    ok = {"libgtsam", "libboost_filesystem", "libtbb", "libmetis-gtsam", "libblas", "libz", "libX11", "libGLX",
          "libbsd", "libopencv_core", "libyaml-cpp", "libatlas", "liblapack", "libstdc++", "libc", "ld-linux-x86-64",
          "libstrataslam_core", "libxcb", "libXau", "libXdmcp", "libGLdispatch", "libmd", "libtbbmalloc"}
    assert audit_mod.ldd_problems(entries, ok) == []
    problems = audit_mod.ldd_problems(entries, ok | {"libsomething_gpl"})
    assert problems == ["unregistered linked library 'libsomething_gpl'"]
    problems = audit_mod.ldd_problems(entries, ok - {"libtbb", "libtbbmalloc"})
    assert problems == ["oneTBB: marked transitive but not loaded by the checked artifacts"]


def test_repository_licences_are_all_accepted() -> None:
    # The four questions of the 2026-11-04 audit (METIS, Spectra, libpng, libjpeg-turbo) were decided in ADR-0051.
    assert run(REPO) == []


def test_weak_copyleft_only_for_transitive_or_embedded_libraries(tmp_path: Path) -> None:
    def problems(name: str, lic: str, status: str) -> list[str]:
        (tmp_path / name).mkdir()
        return run(make_repo(tmp_path / name, [f"| X | libx | 1 | {lic} | u | s | {status} |"]))

    metis = "Apache-2.0 AND LGPL-2.1-or-later (GKlib/getopt.c, from glibc)"
    assert problems("a", metis, "transitive") == []                          # loaded through GTSAM's package
    assert problems("b", "MPL-2.0", "embedded") == []                        # compiled into another library
    assert any("forbidden license 'LGPL-2.1-or-later'" in p for p in problems("c", metis, "active"))  # ours: never
    assert any("only allowed for ['Eigen']" in p for p in problems("d", "MPL-2.0", "active"))
    # Strong copyleft stays forbidden however the library arrives.
    for i, lic in enumerate(["GPL-2.0", "AGPL-3.0", "Apache-2.0 AND GPL-3.0-or-later"]):
        for status in ("transitive", "embedded"):
            assert any("forbidden" in p for p in problems(f"e{i}{status}", lic, status)), (lic, status)


def test_permissive_equivalents_for_any_dependency(tmp_path: Path) -> None:
    root = make_repo(tmp_path, ["| libpng | PNG | 1.6 | libpng-2.0 | u | apt | active |",
                                "| libjpeg-turbo | JPEG | 2.1 | IJG AND BSD-3-Clause AND Zlib | u | apt | active |"],
                     "find_package(PNG)\nfind_package(JPEG)")
    assert run(root) == []


def test_embedded_rows_are_licence_checked_but_need_not_be_loaded(tmp_path: Path) -> None:
    root = make_repo(tmp_path, ["| Inner | libinner | 1 | BSD-3-Clause | u | s | embedded |",
                                "| Gpl | libgplinner | 1 | GPL-2.0 | u | s | embedded |"])
    problems = run(root)
    assert problems == ["Gpl: forbidden license 'GPL-2.0'"]
    entries = audit_mod.parse_registry(root / "THIRD_PARTY.md")
    assert audit_mod.ldd_problems(entries, set()) == []           # not loaded: fine for embedded code
    assert audit_mod.ldd_problems(entries, {"libinner"}) == []    # loaded: registered


def test_quadmath_is_not_exempted_as_toolchain_runtime(tmp_path: Path) -> None:
    entries = audit_mod.parse_registry(make_repo(tmp_path, []) / "THIRD_PARTY.md")
    assert audit_mod.ldd_problems(entries, {"libquadmath", "libgfortran", "libgcc_s"}) == ["unregistered linked library 'libquadmath'"]


def test_unresolved_libraries_are_reported(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class Out:
        stdout = "\tlibfoo.so.1 => not found\n\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x0)\n"
    monkeypatch.setattr(audit_mod.subprocess, "run", lambda *a, **k: Out())
    unresolved: list[str] = []
    libs = audit_mod.linked_libraries(Path("artifact"), unresolved)
    assert "libfoo" in libs and len(unresolved) == 1 and "libfoo.so.1" in unresolved[0]
