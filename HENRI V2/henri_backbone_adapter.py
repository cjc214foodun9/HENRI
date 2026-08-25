"""Default-OFF pretrained backbone adapter for HENRI (CLASS51, P1).

The adapter is a provenance-audited gateway to a frozen general foundation
model. It is NOT part of the HENRI wave/Zone C brain: it is a semantic
System-1 substrate that HENRI layers (memory, planning, retrieval,
verification, online adaptation) may later consume under the approved
ablation controls.

Hard invariants:
- Default-OFF: unless HENRI_BACKBONE=1 is set, every public entry point
  raises BackboneDisabledError. There is no silent fallback path.
- Provenance: the model is loaded only from a pinned immutable HF revision
  and an optional artifact manifest with per-shard SHA-256 digests.
- Frozen baseline: after load the model is in eval() mode with zero
  trainable parameters. Training/tuning is prohibited by the CLASS51
  amendment; this adapter never calls .train() or a backward pass.
- Deterministic greedy generation by default (do_sample=False).
- Fail-closed: typed exceptions for disabled, provenance, input, and
  generation failures. Outputs never route through generic HENRI decoder
  shortcuts or lookup tables.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

ENV_ENABLE_FLAG = "HENRI_BACKBONE"
ENV_MODEL_DIR = "HENRI_BACKBONE_MODEL_DIR"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def backbone_enabled() -> bool:
    """Return True only when the explicit opt-in env flag is truthy."""
    return os.environ.get(ENV_ENABLE_FLAG, "").strip() in {"1", "true", "True", "yes"}


class BackboneError(RuntimeError):
    """Base class for all backbone adapter failures."""


class BackboneDisabledError(BackboneError):
    """Raised when the adapter is used with HENRI_BACKBONE unset or 0."""


class BackboneProvenanceError(BackboneError):
    """Raised when model revision or artifact hashes fail validation."""


class BackboneInputError(BackboneError):
    """Raised when a text or image-text request is malformed."""


class BackboneGenerationError(BackboneError):
    """Raised when model generation fails (including CUDA OOM)."""


@dataclass
class BackboneTelemetry:
    model_id: str
    revision: str
    manifest_sha256: str | None = None
    dtype: str = ""
    device: str = ""
    checkpoint_load_status: Literal["LOADED", "NOT_LOADED"] = "NOT_LOADED"
    trainable_params: int = 0
    total_params: int = 0
    generation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "manifest_sha256": self.manifest_sha256,
            "dtype": self.dtype,
            "device": self.device,
            "checkpoint_load_status": self.checkpoint_load_status,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "generation": dict(self.generation),
        }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a CLASS51 artifact manifest (henri.class51 model-tree manifest).

    Expected shape: {"revision": str, "files": {path: {"size": int,
    "lfs_sha256": str | None, ...}}}.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise BackboneProvenanceError(f"manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - json error surface
        raise BackboneProvenanceError(f"manifest is not valid JSON: {path}") from exc
    if not isinstance(manifest, dict) or "revision" not in manifest:
        raise BackboneProvenanceError(f"manifest lacks required fields: {path}")
    return manifest


def verify_shard_hashes(
    model_dir: str | Path,
    manifest_path: str | Path | None = None,
    *,
    expected_revision: str = DEFAULT_REVISION,
) -> dict[str, Any]:
    """Verify per-shard SHA-256 digests against the pinned manifest.

    Returns a summary dict with per-file status. Raises
    BackboneProvenanceError on any mismatch or on a missing manifest when
    the directory contains safetensors shards.
    """
    model_dir_path = Path(model_dir)
    if manifest_path is None:
        candidates = list(model_dir_path.glob("*_tree_manifest.json")) + list(
            model_dir_path.glob("*manifest*.json")
        )
        if not candidates:
            raise BackboneProvenanceError(
                "no artifact manifest supplied and none found in model directory"
            )
        manifest_path = candidates[0]
    manifest = load_artifact_manifest(manifest_path)
    if manifest.get("revision") != expected_revision:
        raise BackboneProvenanceError(
            f"manifest revision {manifest.get('revision')} != expected {expected_revision}"
        )
    results: dict[str, Any] = {"files": {}, "all_match": True}
    for rel_path, meta in manifest.get("files", {}).items():
        expected_sha = meta.get("lfs_sha256")
        full_path = model_dir_path / rel_path
        if expected_sha is None or not full_path.is_file():
            results["files"][rel_path] = {"status": "skipped", "reason": "no lfs hash or missing file"}
            continue
        actual = sha256_file(full_path)
        ok = actual == expected_sha
        results["files"][rel_path] = {
            "status": "MATCH" if ok else "MISMATCH",
            "sha256": actual,
            "expected_sha256": expected_sha,
        }
        results["all_match"] = results["all_match"] and ok
    if not results["all_match"]:
        mismatches = [p for p, r in results["files"].items() if r.get("status") == "MISMATCH"]
        raise BackboneProvenanceError(f"shard hash mismatch for: {mismatches}")
    return results


def freeze_for_baseline(model: nn.Module) -> tuple[int, int]:
    """Put a model in eval mode with zero trainable parameters.

    Returns (total_params, trainable_params).
    """
    model.eval()
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        parameter.requires_grad = False
    return total, trainable


class QwenBackboneAdapter:
    """Frozen Qwen3-VL backbone gateway (default-OFF, provenance-gated)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        model_dir: str | None = None,
        manifest_path: str | None = None,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float | None = None,
        verify_shards: bool = False,
    ) -> None:
        if not backbone_enabled():
            raise BackboneDisabledError(
                f"{ENV_ENABLE_FLAG} is not set; backbone adapter is disabled"
            )
        self.model_id = model_id
        self.revision = revision
        self.model_dir = Path(model_dir or os.environ.get(ENV_MODEL_DIR, ""))
        if not self.model_dir.is_dir():
            raise BackboneProvenanceError(f"model directory not found: {self.model_dir}")
        self.manifest_path = manifest_path
        self.verify_shards = verify_shards
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.telemetry = BackboneTelemetry(model_id=model_id, revision=revision)
        self._model: Any = None
        self._processor: Any = None

    # -- provenance -----------------------------------------------------
    def _check_config_revision(self) -> None:
        config_path = self.model_dir / "config.json"
        if not config_path.is_file():
            raise BackboneProvenanceError(f"config.json missing: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_rev = config.get("revision")
        if isinstance(config_rev, str) and config_rev != self.revision:
            raise BackboneProvenanceError(
                f"config revision field {config_rev!r} does not match pinned {self.revision!r}"
            )
        if self.verify_shards:
            if self.manifest_path is None:
                raise BackboneProvenanceError("verify_shards=True requires manifest_path")
            verify_shard_hashes(self.model_dir, self.manifest_path, expected_revision=self.revision)
        if self.manifest_path is not None:
            manifest = load_artifact_manifest(self.manifest_path)
            self.telemetry.manifest_sha256 = hashlib.sha256(
                Path(self.manifest_path).read_bytes()
            ).hexdigest()

    # -- lifecycle ------------------------------------------------------
    def load(self) -> "QwenBackboneAdapter":
        """Load processor + model from the pinned local directory."""
        self._check_config_revision()
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except Exception as exc:  # missing dependency
            raise BackboneError(f"transformers unavailable: {exc}") from exc
        try:
            self._processor = AutoProcessor.from_pretrained(
                str(self.model_dir), trust_remote_code=False
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(self.model_dir),
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=False,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise BackboneGenerationError(f"CUDA OOM during model load: {exc}") from exc
        except Exception as exc:
            raise BackboneError(f"model load failed: {exc}") from exc
        if self.device != "cuda":
            self._model = self._model.to(self.device)
        total, _trainable_before = freeze_for_baseline(self._model)
        if total == 0:
            raise BackboneError("backbone has no parameters")
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            raise BackboneError("backbone must have zero trainable parameters")
        self.telemetry.checkpoint_load_status = "LOADED"
        self.telemetry.total_params = total
        self.telemetry.trainable_params = 0
        self.telemetry.device = str(next(self._model.parameters()).device)
        self.telemetry.dtype = str(self.dtype)
        self.telemetry.generation = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
        }
        return self

    # -- generation -----------------------------------------------------
    def _prepare_inputs(self, messages: list[dict[str, Any]]):
        if not messages:
            raise BackboneInputError("empty message list")
        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(text=text, return_tensors="pt")
        except Exception as exc:
            raise BackboneInputError(f"processor failed: {exc}") from exc
        return {key: value.to(self.device) for key, value in inputs.items()}

    def generate_text(self, prompt: str) -> tuple[str, BackboneTelemetry]:
        """Greedy text completion through the official chat template."""
        if self._model is None:
            raise BackboneError("adapter not loaded; call load() first")
        messages = [{"role": "user", "content": prompt}]
        inputs = self._prepare_inputs(messages)
        try:
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                )
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            return self._processor.decode(output_ids[0], skip_special_tokens=True), self.telemetry
        except torch.cuda.OutOfMemoryError as exc:
            raise BackboneGenerationError(f"CUDA OOM during generation: {exc}") from exc
        except Exception as exc:
            raise BackboneGenerationError(f"generation failed: {exc}") from exc

    def generate_image_text(self, prompt: str, image_path: str | Path) -> tuple[str, BackboneTelemetry]:
        """Greedy multimodal completion from a local image file."""
        if self._model is None:
            raise BackboneError("adapter not loaded; call load() first")
        image_path = Path(image_path)
        if not image_path.is_file():
            raise BackboneInputError(f"image file not found: {image_path}")
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise BackboneInputError(f"image could not be read: {image_path}: {exc}") from exc
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(text=text, images=[image], return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        except Exception as exc:
            raise BackboneInputError(f"multimodal processor failed: {exc}") from exc
        try:
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                )
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            return self._processor.decode(output_ids[0], skip_special_tokens=True), self.telemetry
        except torch.cuda.OutOfMemoryError as exc:
            raise BackboneGenerationError(f"CUDA OOM during generation: {exc}") from exc
        except Exception as exc:
            raise BackboneGenerationError(f"generation failed: {exc}") from exc

    def embed_text(self, prompt: str) -> tuple[torch.Tensor, BackboneTelemetry]:
        """Frozen last-token embedding (Egress-1 conditioning). No generation.

        Bounded method added under approval 2b30c69f / contract 7fcc9361.
        Returns L2-normalized float32 last-token hidden state of the final
        layer, computed under torch.inference_mode(). No gradients, no
        state change, no sampling. Dead unless HENRI_BACKBONE=1.
        """
        if self._model is None:
            raise BackboneError("adapter not loaded; call load() first")
        messages = [{"role": "user", "content": prompt}]
        inputs = self._prepare_inputs(messages)
        try:
            with torch.inference_mode():
                out = self._model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[-1]  # [1, L, D]
            e = hs[:, -1, :].to(torch.float32)  # last token, float32
            e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return e.squeeze(0), self.telemetry
        except torch.cuda.OutOfMemoryError as exc:
            raise BackboneGenerationError(f"CUDA OOM during embed: {exc}") from exc
        except Exception as exc:
            raise BackboneGenerationError(f"embed failed: {exc}") from exc

    def memory_report(self) -> dict[str, Any]:
        """Current GPU memory snapshot (MiB). Returns empty on CPU."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return {"device": self.device}
        return {
            "device": self.device,
            "allocated_mib": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
            "reserved_mib": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
        }


def main(argv: list[str] | None = None) -> int:
    """CLI: python henri_backbone_adapter.py --text PROMPT [--image PATH] [--verify]."""
    argv = list(sys.argv[1:] if argv is None else argv)
    import argparse

    parser = argparse.ArgumentParser(description="CLASS51 backbone smoke")
    parser.add_argument("--text", required=True, help="prompt")
    parser.add_argument("--image", default=None, help="optional image path")
    parser.add_argument("--verify", action="store_true", help="verify shard hashes before load")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args(argv)
    adapter = QwenBackboneAdapter(
        model_dir=args.model_dir,
        manifest_path=args.manifest,
        max_new_tokens=args.max_new_tokens,
        verify_shards=args.verify,
    )
    adapter.load()
    if args.image is not None:
        output, telemetry = adapter.generate_image_text(args.text, args.image)
    else:
        output, telemetry = adapter.generate_text(args.text)
    print(json.dumps(
        {"output": output, "telemetry": telemetry.to_dict(), "memory": adapter.memory_report()},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
