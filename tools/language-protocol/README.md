# Language infrastructure checks

Run through Hermes terminal from this directory:

```bash
python -m unittest test_language test_language_boundaries test_consumers test_truncation test_stative -v
```

Tests use the declared repo overlay sentinel, or explicit HENRI_LANGUAGE_TEST_CLI in an isolated process.
Consumer imports resolve from that same profile root. No hidden installed-copy substitution occurs.
The suite tests selected regex/count boundaries, exact formal/visual preservation, fixed packet policy, and malformed input refusal.
The hook unit test substitutes Jev only to test the tail wiring. Separate hook-receipt.json records the real Jev call.
The guard unit test blocks before _invoke on malformed description. Actual sandbox readback was not Ready and stopped before proof/dispatch.
No HENRI model, native gRPC middleware, linguistic certification, or end-to-end performance claim follows from these tests.

Rendering uses the installed Draw.io skill. The self-contained HTML report uses creative/claude-design guidance.
The real Edge DOM test covers tabs, 1280/390 layout, row count, overflow, and target size. Visual review has a separate scope.
Do not add bytecode, browser profiles, screenshots, official PDF/dictionary, raw MoA text, private TLS keys, or API responses to Git.
