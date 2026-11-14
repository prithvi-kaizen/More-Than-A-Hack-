# Working instructions

Do not use `codex` anywhere in new Git branch names.

Implement one phase per user authorization. After each phase, verify results,
update `docs/PHASES.md` and the implementation plan, commit and push to the
configured GitHub repository, then stop. Wait for the user to say continue.

Restricted inputs and derived artifacts remain local. Do not display their contents,
plots, predictions or detailed logs in tool output. Use synthetic fixtures for debugging
and fixed PASS/FAIL statuses for target-data checks. Train and tune only on external
matches. Never select a model using the competition public labels or target video.

Run `uv run --frozen python -m unittest discover -s tests -v` and
`uv run --frozen python scripts/check_staged_files.py` before each phase push.
