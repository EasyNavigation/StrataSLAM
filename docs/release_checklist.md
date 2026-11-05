# Release checklist

What was verified in the development environment is in [results.md](results.md), [scenarios.md](scenarios.md) and
[benchmarks.md](benchmarks.md). These steps need accounts and services outside the repository and are
the maintainer's:

- [ ] Push the repository to its public home and let the CI run there
      (the workflows were executed locally piece by piece, in clean containers, but never on GitHub).
- [ ] Decide the version (the `[Unreleased]` section of `CHANGELOG.md`), move it to a dated section, and create a
      signed tag (`git tag -s vX.Y.Z`).
- [ ] Connect the repository to Zenodo and publish the release (`.zenodo.json` has the metadata); copy the DOI into
      `CITATION.cff` (`doi:`) and into the README.
- [ ] Run the nightly workflow once (`workflow_dispatch`) on the release commit: `scripts/make_results.sh`
      regenerates `docs/results.md` and `docs/scenarios.md`; commit them with the tag.
- [ ] Decide the open items marked `TODO(decision)`: reference LiDAR, code style. (The project license is
      Apache-2.0 and the licences of the dependencies are decided: ADR-0051.)
