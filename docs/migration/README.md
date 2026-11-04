# Migration records

These reports document how py3wafo was ported from MATLAB WAFO 2.1.1 and
how each function was validated. Users of the library do not need them; they
are kept for transparency and for contributors.

| Report | Contents |
|---|---|
| [FUNCTION_VALIDATION.md](FUNCTION_VALIDATION.md) | Summary of the latest validation run |
| [PASSED_FUNCTIONS.md](PASSED_FUNCTIONS.md) | Every public function and the tests that exercise it |
| [FAILED_FUNCTIONS.md](FAILED_FUNCTIONS.md), [UNTESTED_FUNCTIONS.md](UNTESTED_FUNCTIONS.md) | Functions touched by failing tests, or without tests (both empty at 0.4.0) |
| [UNIMPLEMENTED_FUNCTIONS.md](UNIMPLEMENTED_FUNCTIONS.md) | MATLAB functions without a Python counterpart (none at 0.4.0) |
| [UNREVIEWED_SCRIPTS.md](UNREVIEWED_SCRIPTS.md) | Standalone MATLAB scripts not yet ported |
| [FUNCTION_CATALOG.md](FUNCTION_CATALOG.md), [conversion_manifest.csv](conversion_manifest.csv) | Every MATLAB file and its Python counterpart |
| [COMPATIBILITY.md](COMPATIBILITY.md) | API and behaviour differences from MATLAB |
| [REPAIR_LOG.md](REPAIR_LOG.md) | Defects found during validation, the fixes and their evidence |
| [TOOLBOX_STRUCTURE.md](TOOLBOX_STRUCTURE.md) | Structure of the original MATLAB toolbox |
| [NATIVE_DEPENDENCIES.md](NATIVE_DEPENDENCIES.md), [NATIVE_PYTHON_PORTS.md](NATIVE_PYTHON_PORTS.md), [non_matlab_files.csv](non_matlab_files.csv) | The original C/Fortran/binary files and their Python replacements |
| [REVIEWED_SOURCE_DOCUMENTATION.md](REVIEWED_SOURCE_DOCUMENTATION.md) | Original help pages available through `wafo.docs.wafohelp` |
| [CHECKPOINT_0_3.md](CHECKPOINT_0_3.md), [CHECKPOINT_0_4.md](CHECKPOINT_0_4.md), [VALIDATION.md](VALIDATION.md) | Historical validation checkpoints |

Some reports quote paths from the development workspace (for example
`../wafo_matlab` or `dist/`); they describe the checkpoint at which they were
written.

## Regenerating the reports

```bash
python -m pip install -e ".[test]"
python tools/validate_functions.py                     # tests + function reports
python tools/validate_functions.py --require-complete  # also fail on missing evidence
```

`tools/audit_toolbox.py` rebuilds the source inventories. It needs a copy of
the original MATLAB toolbox, by default in `../wafo_matlab` next to this
repository: `python tools/audit_toolbox.py path/to/wafo_matlab`.
