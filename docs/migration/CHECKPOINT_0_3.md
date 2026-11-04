# py3wafo 0.3.0 validation checkpoint

This is a tested migration checkpoint, **not a completed WAFO toolbox port**.

* Source regression suite: **415 passed**, no failed or skipped tests.
* Independently installed wheel regression suite: **415 passed**, with source
  import paths disabled and coefficient tables loaded from the installed wheel.
* Public API names with actual passing-test execution evidence: **445**.
  Aliases/re-exports share evidence; this is not a count of unique algorithms.
* Public API names touched by failing tests: **0**; untested exposed APIs: **0**.
* MATLAB source coverage: **225 implemented**, **182 adapted**, **276 pending**.
* Of 79 source documentation/script records, **42 scripts** still need review
  or porting; they are not silently considered completed documentation.

The completion-gated validation command returns **exit status 1**, despite all
415 regression tests passing, because unresolved source functions/scripts remain.

## Requested lists and repair evidence

* [Passed functions and their tests](PASSED_FUNCTIONS.md)
* [Current failed-function evidence](FAILED_FUNCTIONS.md)
* [Untested APIs](UNTESTED_FUNCTIONS.md)
* [Unimplemented functions and reasons](UNIMPLEMENTED_FUNCTIONS.md)
* [Source scripts still requiring review](UNREVIEWED_SCRIPTS.md)
* [Repair log](REPAIR_LOG.md)
* [Recorded test/repair iterations and failure tracebacks](validation_history.json)
* [Machine-readable function results](function_test_results.csv)
* Source and installed-wheel JUnit results: `test_results.xml`,
  `wheel_test_results.xml`.

## Deliverable

`../dist/py3wafo-0.3.0-py3-none-any.whl` contains both `py3wafo` and compatible
`wafo` import namespaces, measurement data, native capability mappings, and
numeric-only empirical model tables with original-file hashes/provenance.
No original WAFO executable is loaded. Format converters remain optional
external tools, and scientific method/option limitations are documented in
the function docstrings and compatibility report.

Reproduce the source audit and tests:

```powershell
python tools/audit_toolbox.py ..\wafo_matlab
python tools/validate_functions.py
python tools/validate_functions.py --require-complete
```

The last command is expected to fail the completion gate at this checkpoint.
The original `wafo_matlab` source tree remains unchanged.
