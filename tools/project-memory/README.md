# Project-memory infrastructure checks

These stdlib unittest checks are separate from HENRI model tests. They exercise source/content hash binding, cross-process recall, refusal cases, and a real temporary local Git remote. The tiny local Git fixture is labeled as a plumbing fixture, not a HENRI result. No Honcho client/server/model is simulated.

Run through Hermes `terminal` from `tools/project-memory`:

```bash
python -m unittest test_project_memory test_memory_guards additional_guards test_bounded_records test_review_findings -v
```

The tests resolve the repo from the committed overlay sentinel and test that overlay. Standalone testing supports explicit `HENRI_PROJECT_MEMORY_TEST_REPO` and `HENRI_PROJECT_MEMORY_TEST_CLI` process environment values; these are test plumbing, not persistent Hermes settings or credentials. The source fixture SHA is pinned so tests still address the same real source after a new commit. Ensure that Git object exists in a full fetch. Native Windows junction control uses `cmd /c mklink /J`; other hosts use a real symlink. No test requires HENRI GPU libraries.

The full run has inherited repeat executions. Read `test_executions` and `unique_test_names` separately; neither is a model-capability score. Output artifacts stay outside Git. The real-remote test disables bytecode while importing the overlay so it does not contaminate its exact-file manifest.

Limits: known lexical patterns are not comprehensive data-loss prevention; explicit public-review acknowledgement is caller intent, not verified human identity. Tests do not prove concurrent hostile filesystem safety. Final actual GitHub ref/blob readback is a separate delivery check. Every test executes against temporary local stores except read-only immutable HENRI Git source objects.
