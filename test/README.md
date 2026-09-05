# test/

- **`unit/`** — automated pytest suite for `AtomicSystem` (all mixins: core
  properties, `EditMixin`, `TopologyMixin`, `ForceFieldMixin`, `IOMixin`,
  and the packmol-backed `add_structure`/`add_liquid_layer`/`add_droplet`).
  This is the only directory tracked in git (see `.gitignore`). Run with:

  ```
  pytest
  ```

  from the repo root (needs a full runtime env with MDAnalysis, pymatgen,
  rdkit, packmol on PATH — e.g. the `cemd_ui` conda env; `pyproject.toml`'s
  `[tool.pytest.ini_options]` restricts collection to `test/unit`).
  `add_liquid_layer`/`add_droplet` tests are skipped automatically if the
  `packmol` binary isn't on PATH.

- **`legacy/`** — pre-existing manual/example scripts and sample structure
  files, kept for reference but not run automatically. `legacy/build_old/`
  is a superseded build-system implementation (replaced by `cemd/build/`).

- **`tutorials/`** — worked examples with their output data/figures
  (`caffeine/`, `slab_caco3/`).
