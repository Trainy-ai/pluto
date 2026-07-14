CI Tests for the MLOp client are done against the production instance of
MLOp server. The default urls for each api endpoint is listed in
`mlop/sets.py`

## Test organization

Group tests by the functionality under test, not by test style. When adding
tests for an area that an existing file already covers, extend that file rather
than creating a new one — keep unit tests and their end-to-end/integration
counterparts together so the full coverage for a feature lives in one place.

For example, everything exercising process signal handling and shutdown
(SIGTERM/SIGINT behavior, the SIGTERM→TERMINATED status handler, graceful
`finish()`/`close()` on exit) lives in `test_shutdown.py` — both the direct
unit tests and the subprocess integration tests that raise real signals.
Prefer adding a class or a helper-script + test there over a new
`test_sigterm_*.py` / `test_signal_*.py` file.
