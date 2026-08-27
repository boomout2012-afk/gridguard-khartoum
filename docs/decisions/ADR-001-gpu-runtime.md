# ADR-001: Use a fail-closed PyTorch and CuPy CUDA runtime

## Status

Accepted

## Date

2026-08-15

## Context

GridGuard must replay batched 24-hour scenarios, run the trained PPO controller, calculate evaluation results, and remain responsive while a native desktop presentation is active. The presentation laptop has an NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB of memory, a CUDA-compatible driver, and a verified PyTorch 2.8 CUDA 12.8 installation. CPU execution is a significant bottleneck for the intended benchmark and presentation workflow.

The accepted source implementation is already tensor-oriented and its trained policy is a PyTorch network. The clean application also needs efficient numerical and statistical operations without repeatedly transferring arrays through host memory.

## Decision

- Require a compatible NVIDIA GPU for the primary desktop demonstration and benchmark workflow.
- Run PPO inference and the batched simulator with PyTorch 2.8 on `cuda:0`.
- Use CuPy for GPU-native numerical and statistical operations.
- Exchange tensors between PyTorch and CuPy through DLPack only at explicit module boundaries so compatible operations remain zero-copy.
- After one explicit packed device-to-host transfer, use NumPy/Python only for deterministic
  canonical row, cell, aggregate, and manifest assembly plus small host-side validation. This is
  a serialization boundary, not a controller or benchmark-compute fallback.
- Run a fail-closed launch preflight that verifies CUDA availability, device identity, both libraries, and coherent DLPack sharing. Do not silently switch the presentation or benchmark to CPU.
- Permit small CPU unit tests through explicit device injection. The portable command is:

  ```powershell
  uv run --no-sync pytest -m "not gpu and not source_archive and not release_artifact"
  ```

- Separate release assurance into the mandatory presentation-GPU gate
  (`uv run --no-sync pytest -ra -m "not source_archive and not release_artifact"`), the strict
  post-build `release_artifact` gate (`uv run --no-sync pytest -ra -m "release_artifact"`), and the
  optional source-archive parity gate. The normal package build does not require the sibling
  research archive.

## Alternatives considered

### CPU-only PyTorch

- Pros: smaller operational surface and easier execution on generic machines.
- Cons: does not meet the required presentation performance and contradicts the approved hardware direction.
- Rejected: the project is being finalized for a known NVIDIA presentation laptop, and CPU throughput is inadequate for the intended workflow.

### Export the actor to a NumPy-only runtime

- Pros: substantially smaller package and no CUDA dependency.
- Cons: abandons the requested GPU path, adds a second inference implementation, and creates another numerical parity boundary.
- Rejected: packaging size does not outweigh the performance and parity requirements.

### Use only CuPy

- Pros: one NumPy-like GPU array library.
- Cons: the trained PPO policy is a PyTorch network; reimplementing the model would increase migration and validation risk.
- Rejected: PyTorch remains the authoritative policy runtime.

### Use only PyTorch

- Pros: one GPU memory manager and fewer dependencies.
- Cons: the project requires CuPy for the dedicated GPU-native statistics path and NumPy-compatible analysis operations.
- Rejected: the two-library design is acceptable when the boundary is explicit and DLPack avoids host copies.

## Consequences

- The presentation package is larger and requires an NVIDIA-compatible installation.
- Startup can fail when CUDA, PyTorch, CuPy, the driver, or DLPack interoperability is unavailable; the diagnostic must identify the failed layer clearly.
- CUDA work is asynchronous, so performance measurements must synchronize with CUDA events or an explicit device synchronization.
- Tensor ownership and stream behavior must be documented at each DLPack boundary.
- CI without an NVIDIA runner may execute the explicitly filtered CPU-safe unit tests, but it
  cannot certify a release. Mandatory runtime verification occurs on the presentation GPU with no
  selected test skips, and the final Windows tree must separately pass every `release_artifact`
  test. Source-archive parity remains optional and explicit.
- The application remains fully offline after dependencies and artifacts are installed.
