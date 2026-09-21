from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from inference_planner.core.enums import Confidence
from inference_planner.validation.registry import get_validator, register_validator
from inference_planner.validation.vllm_probe import (
    VLLMSubprocessValidator,
    _parse_last_json_line,
)
from tests.conftest import make_hardware, make_model


def _fake_process(stdout: str, returncode: int = 0) -> MagicMock:
    return MagicMock(stdout=stdout, stderr="", returncode=returncode)


def test_validate_returns_measured_result_on_success():
    validator = VLLMSubprocessValidator()
    stdout = '{"success": true, "load_time_seconds": 12.5, "vram_gb": 20.1, "tokens_per_second": 30.0}'
    with patch("subprocess.run", return_value=_fake_process(stdout)):
        result = validator.validate(make_model(), make_hardware(gpu_count=1), {"model": "x"})

    assert result.confidence == Confidence.MEASURED
    assert result.model_load_time_seconds == 12.5
    assert result.actual_vram_usage_gb == 20.1
    assert result.tokens_per_second == 30.0


def test_validate_returns_unknown_on_timeout():
    validator = VLLMSubprocessValidator()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=5)):
        result = validator.validate(make_model(), make_hardware(gpu_count=1), {"model": "x"}, timeout_seconds=5)

    assert result.confidence == Confidence.UNKNOWN
    assert any("timed out" in note for note in result.notes)


def test_validate_returns_unknown_when_probe_reports_failure():
    validator = VLLMSubprocessValidator()
    stdout = '{"success": false, "error": "CUDA out of memory"}'
    with patch("subprocess.run", return_value=_fake_process(stdout, returncode=1)):
        result = validator.validate(make_model(), make_hardware(gpu_count=1), {"model": "x"})

    assert result.confidence == Confidence.UNKNOWN
    assert any("CUDA out of memory" in note for note in result.notes)


def test_validate_returns_unknown_when_output_unparseable():
    validator = VLLMSubprocessValidator()
    with patch("subprocess.run", return_value=_fake_process("garbage, not json", returncode=1)):
        result = validator.validate(make_model(), make_hardware(gpu_count=1), {"model": "x"})

    assert result.confidence == Confidence.UNKNOWN


def test_validate_returns_unknown_when_no_config_given():
    validator = VLLMSubprocessValidator()
    result = validator.validate(make_model(), make_hardware(gpu_count=1), None)
    assert result.confidence == Confidence.UNKNOWN


def test_validate_reports_partial_success_when_generation_fails():
    validator = VLLMSubprocessValidator()
    stdout = '{"success": true, "load_time_seconds": 5.0, "vram_gb": 10.0, "tokens_per_second": null, "generation_error": "boom"}'
    with patch("subprocess.run", return_value=_fake_process(stdout)):
        result = validator.validate(make_model(), make_hardware(gpu_count=1), {"model": "x"})

    assert result.confidence == Confidence.MEASURED
    assert result.model_load_time_seconds == 5.0
    assert result.tokens_per_second is None
    assert any("generation failed" in note for note in result.notes)


def test_parse_last_json_line_picks_the_last_valid_line():
    stdout = "some log noise\nmore noise\n" + '{"success": true, "load_time_seconds": 1.0}'
    assert _parse_last_json_line(stdout) == {"success": True, "load_time_seconds": 1.0}


def test_parse_last_json_line_returns_none_for_no_json():
    assert _parse_last_json_line("nothing but logs here") is None


def test_validator_registry_lookup():
    assert get_validator("vllm") is not None
    assert get_validator("does-not-exist") is None


def test_validator_registry_register_and_lookup():
    class _DummyValidator:
        def validate(self, *a, **kw):
            raise NotImplementedError

    dummy = _DummyValidator()
    register_validator("dummy-runtime", dummy)
    assert get_validator("dummy-runtime") is dummy
