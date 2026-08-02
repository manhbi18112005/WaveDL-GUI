# WaveDL Qt Desktop Improvement Design

**Status:** user-approved direction; implementation specification
**Date:** 2026-08-02
**Decision:** retain PySide6 Widgets and qfluentwidgets. Do not migrate to Electron, Tauri, or QML in this phase.

## 1. Goals and constraints

The GUI remains a native Qt Widgets shell, while training, testing, and plot generation run in a separately managed WaveDL runtime. The boundary must be usable from a terminal, independent of shell formatting, and stable across UI releases. It must support Windows, macOS, and Linux; CPU, Apple MPS, and NVIDIA CUDA; and users who already maintain a compatible Python environment.

The current `TrainingWorker`, `TestWorker`, and `PlotWorker` in `src/wavedl_gui/service/training_service.py` launch modules with `get_python_executable()` (currently `sys.executable`), merge stdout and stderr, parse the legacy `##METRICS##` prefix, and use POSIX process-group calls for cancellation. This design replaces those assumptions without changing the Qt shell technology.

## 2. Runtime boundary and JSONL protocol

### 2.1 Contract

The runtime emits one UTF-8 JSON object per line to **stdout**. Each object has:

```json
{"protocol":"wavedl-jsonl","version":1,"run_id":"uuid","type":"metric","seq":7,"ts":"2026-08-02T12:00:00Z","payload":{}}
```

Required envelope fields:

- `protocol`: exactly `wavedl-jsonl`.
- `version`: integer protocol major version; incompatible changes require a new major.
- `run_id`: stable UUID for the invocation.
- `type`: one of `hello`, `state`, `metric`, `log`, `artifact`, `warning`, `error`, `exit`.
- `seq`: monotonically increasing integer starting at zero for each run.
- `ts`: UTC RFC 3339 timestamp.
- `payload`: object whose schema depends on `type`.

Unknown fields must be ignored. Unknown event types are logged as protocol warnings and do not terminate a run. A malformed line is retained in diagnostics, reported as a GUI warning, and does not prevent subsequent valid lines from being consumed. stdout is protocol-only; human-readable diagnostics use `log` events or stderr captured separately.

### 2.2 Event payloads

- `hello`: `{ "protocol_min": 1, "protocol_max": 1, "runtime_version": "...", "python": "...", "device": "cpu|mps|cuda", "capabilities": [...] }`. It is the first event; absence within the startup deadline is a protocol failure.
- `state`: `{ "state": "starting|running|stopping|completed|failed|cancelled", "message": "..." }`.
- `metric`: `{ "name": "epoch", "value": ..., "step": 12, "unit": "...", "metrics": { ... } }`. `metrics` carries the existing epoch fields (`train_loss`, `val_loss`, `lr`, `r2`, `pearson`, patience, timing, and optional per-parameter MAE) without requiring the UI to parse text.
- `log`: `{ "stream": "stdout|stderr|system", "level": "debug|info|warning|error", "message": "..." }`.
- `artifact`: `{ "kind": "checkpoint|plot|history|other", "path": "absolute-or-run-relative-path", "sha256": "...", "size": 123, "label": "..." }`. Paths are resolved against the run directory and must not escape it unless explicitly declared by the runtime.
- `warning`: `{ "code": "...", "message": "...", "details": {} }`; non-fatal and surfaced in the activity log.
- `error`: `{ "code": "...", "message": "...", "details": {}, "recoverable": false }`; the GUI shows an actionable error and records the complete event.
- `exit`: `{ "code": 0, "status": "completed|failed|cancelled", "message": "...", "artifacts": [...] }`. The GUI treats this as authoritative only after the subprocess has exited, and verifies the OS return code is consistent.

The GUI sends no protocol commands over stdin. Cancellation is an OS-level supervisor operation, making the protocol shell-independent. A future command channel, if needed, gets a separately versioned contract.

### 2.3 Compatibility and diagnostics

The runtime must emit `hello` before work begins and reject an unsupported protocol version with a structured `error` followed by `exit`. The UI records raw JSONL and stderr in a per-run diagnostics file. Logs are bounded in memory but never silently dropped from disk. Protocol parsing is a pure Python component with no Qt dependency, enabling terminal and unit testing.

## 3. Explicit WaveDL runtime selection

The GUI must never infer the training interpreter from frozen or GUI process state. Replace the current `sys.executable` behavior with an explicit `RuntimeSpec` selected in this order:

1. User-selected runtime path stored in application settings.
2. A validated bundled runtime pack selected by platform and device mode.
3. An explicitly configured environment variable or documented command path.
4. Existing-environment mode, using a user-confirmed Python executable or `wavedl-train`/module probe.

Each candidate is validated by launching a short probe that returns WaveDL version, Python version, protocol version, and device capabilities. The UI stores the canonical executable path, environment mode, runtime-pack identifier, and last validation result. `sys.executable` may be used only for a development fallback when explicitly enabled and visibly labeled; it is not a release default.

The supervisor launches `python -m wavedl.runtime` (or the equivalent installed runtime entry point), not a shell command string. Arguments are an argv list, with a generated run manifest containing config, protocol version, run directory, and requested device. Environment variables such as `CUDA_VISIBLE_DEVICES` are explicit runtime settings rather than implicit mutation; the current non-macOS default of setting it to `0` must not be retained without user-visible policy.

## 4. Process-tree supervision and cancellation

Create one platform-neutral supervisor abstraction used by training, test, and plot runs. It owns the `Popen` handle, reader threads/tasks, deadlines, diagnostics, and final result. It must:

- launch without a shell, with stdin closed and stdout/stderr captured separately;
- create a new process group/session on POSIX (`start_new_session=True`);
- create a Windows Job Object, assign the child immediately, and configure kill-on-job-close;
- terminate the complete tree on cancellation: graceful request first, then a bounded wait, then forceful group/job termination;
- use monotonic deadlines and be idempotent if cancellation races with normal exit;
- emit exactly one terminal result and distinguish `cancelled`, timeout, launch failure, protocol failure, and runtime failure;
- reap all descendants and close pipes before the Qt worker is released.

The UI transitions `starting → running → stopping → cancelled|completed|failed` and disables duplicate starts. Application shutdown calls supervisor cancellation and waits for bounded cleanup before closing the database. A stuck process produces an error with the run directory and recovery instruction, rather than blocking the Qt event loop.

## 5. Configuration and SQLite ownership/migration

### 5.1 Ownership

The **GUI owns presentation preferences**: theme, language, window state, update preference, onboarding state, selected runtime, and non-run UI defaults. The **runtime owns execution configuration**: model, data path, output path, hyperparameters, device, precision, cache policy, and reproducibility metadata. The GUI creates an immutable, versioned run manifest from its form state; the runtime never reads the GUI's qfluentwidgets config file directly.

`src/wavedl_gui/common/config.py` currently persists both UI and training fields through qconfig to `CONFIG_FILE` defined in `common/setting.py`. Migrate training fields to a namespaced `trainingDefaults` document only for form defaults, while each submitted run receives a complete manifest. Preserve existing keys through a one-time migration, write atomically, and retain a migration version. Invalid values are reported and replaced with defaults, never partially applied.

The GUI owns the application SQLite database initialized by `MainWindow.initDatabase()` and accessed through the database service. Runtime workers must not open that database. Run metadata, event index, status, and artifact references are written by the GUI after validating runtime events. Training outputs remain in the user-selected output/run directory and are not copied into SQLite.

### 5.2 Migration rules

On first launch after the change: back up `config.json` and `database.db`, migrate schema/data in a transaction, record `schema_version` and `migration_id`, then reopen through the existing database initialization path. Existing run records must remain readable. A failed migration restores the backup and starts in read-only recovery mode with an actionable error. Do not delete legacy files until a later release has verified successful startup and a user-approved cleanup path.

## 6. Packaging, signing, runtime packs, and updates

Ship one signed UI shell plus separately versioned runtime packs:

- CPU pack for all supported platforms;
- macOS arm64 MPS pack (and a separately tested Intel/CPU option where supported);
- Windows/Linux CUDA packs tied to a documented CUDA/PyTorch compatibility matrix;
- existing-environment mode, which downloads no runtime and requires validation.

The shell package is built from `wavedl-gui.spec`, which currently bundles WaveDL modules and contains stale PyQt6 hidden-import comments despite the PySide6 implementation. The new release build must decide explicitly whether a pack is external or bundled; do not silently produce a partially frozen runtime. Record build inputs, platform, architecture, Python, PyTorch, and protocol versions in a signed manifest with SHA-256 hashes.

CI currently runs Ruff and CPU unit tests on Ubuntu Python 3.11–3.13, while `release.yml` tests and publishes the Python package on tag. Extend release automation with per-platform shell/pack builds, smoke tests, artifact hash generation, and retention of build provenance. Sign Windows binaries, sign and notarize macOS app/installer, and sign Linux packages or archives with the project release key. Never put signing credentials in repository artifacts.

Updates are staged: check a signed manifest, download to a versioned cache, verify hash/signature, install side-by-side, and switch an atomic current pointer. Keep the previous shell and runtime pack until the new version passes startup and protocol probes. On failed launch or explicit rollback, restore the previous pointer and preserve diagnostics. Runtime packs may be updated independently only when their declared protocol range and shell compatibility match.

## 7. Milestones and gates

1. **M1 — Contract and parser:** publish protocol v1 schemas, fixtures, pure parser, run manifest, and compatibility policy. Gate: fixtures cover every event and malformed/unknown input.
2. **M2 — Runtime adapter:** emit events from train/test/plot entry points and provide the hello probe. Gate: terminal invocation works without the GUI and exits with deterministic status.
3. **M3 — Supervisor integration:** replace worker launch/parsing and implement POSIX/Windows tree control. Gate: cancellation kills descendants on all three OS families; no GUI-thread blocking.
4. **M4 — Ownership migration:** migrate config and SQLite schema with backup/rollback. Gate: upgrade from representative legacy data preserves settings and run history.
5. **M5 — Packaging:** build shell and CPU/MPS/CUDA packs plus existing-environment flow. Gate: clean-machine install, signature verification, runtime probe, and one short run per supported matrix.
6. **M6 — Release/update:** staged update, rollback, notarization/signing, and documentation. Gate: failed update automatically returns to the prior known-good version.

## 8. Scope exclusions

No Electron, Tauri, QML, or widget-framework replacement; no visual redesign; no remote execution/HPC scheduler protocol; no database sharing with runtime; no arbitrary plugin execution; no live protocol command channel; no automatic CUDA driver installation; and no promise of binary portability across unsupported architectures. Model/training algorithm changes are out of scope except for structured event emission.

## 9. Error handling and targeted verification

Every failure includes a stable code, user-safe message, technical details in diagnostics, run ID, and recovery action. Distinguish missing executable, incompatible Python/WaveDL, unsupported device, invalid manifest, permission/disk failure, malformed protocol, timeout, cancellation, non-zero runtime exit, and update signature/hash failure. `MainWindow` continues to surface user-facing errors through the existing signal path, but raw tracebacks remain out of the notification.

Targeted tests: protocol schema/ordering/unknown-event/malformed-line tests; parser backpressure and large-log tests; runtime probe and explicit-interpreter tests; manifest redaction and path-containment tests; supervisor cancellation, timeout, signal-race, and descendant tests on Windows/macOS/Linux; config migration backup/rollback tests; SQLite transaction and concurrent-read tests; artifact hash tests; and Qt integration tests proving start, progress, warning, completion, failure, cancellation, shutdown, and duplicate-start behavior.

Release verification must run the signed shell with each pack on clean machines, validate CPU/MPS/CUDA capability reporting, execute a short deterministic training and plot run, inspect persisted config/database migrations, exercise offline existing-environment mode, verify update signature/hash rejection, and perform rollback after a deliberately broken release. CI should retain JSONL diagnostics and build manifests for failed matrix jobs.

## 10. In-place self-review

- **No placeholder contract:** event names, required envelope fields, terminal semantics, interpreter selection, ownership, and platform process rules are explicit.
- **No contradiction:** stdout is JSONL-only while stderr is separately captured; cancellation is OS-level because stdin is intentionally not a command channel.
- **No frozen-runtime ambiguity:** the shell and runtime packs are separate release units, and `sys.executable` is not a production default.
- **No data-ownership ambiguity:** qconfig remains GUI-owned, manifests/runtime execution are run-owned, and SQLite remains GUI-owned.
- **No unsupported promise:** CUDA/MPS availability is validated by the selected pack and host driver; unsupported combinations fail before launch with a structured diagnostic.
