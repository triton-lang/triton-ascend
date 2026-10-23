import json
from types import SimpleNamespace

import pytest

from triton.backends.ascend.compiler import _update_dynamic_cv_costmodel_report

pytestmark = pytest.mark.backend("cpu")


def metadata(effective):
    report = {
        "selected_route_elapsed_prediction": {"predicted_microseconds": 7.0},
        "loop_sensitivity": {
            "loops": [{
                "stage_dependency_elapsed_prediction": {
                    "status": "requested_on_requires_backend_confirmation",
                    "dynamic_cv_requested": True,
                    "backend_effective_dynamic_cv_confirmed": False,
                    "offline_schedule_calibration_certified": True,
                    "schedule_contract": "test_contract_v1",
                    "full_kernel_validation_sha256": "a" * 64,
                }
            }]
        },
    }
    return {
        "enable_dynamic_cv_pipeline": effective,
        "auto_simt_scope_report": json.dumps(report),
    }


@pytest.mark.parametrize(
    "effective,rc,readable,compile_on_910_95,state,status,confirmed",
    [
        (
            True,
            -1,
            True,
            True,
            "pass_completed",
            "qualified_on_full_tile_candidate",
            True,
        ),
        (
            False,
            1,
            True,
            True,
            "fell_back",
            "requested_on_rejected_after_backend_fallback",
            False,
        ),
        (
            True,
            -1,
            False,
            True,
            "unverified",
            "requested_on_requires_backend_confirmation",
            False,
        ),
    ],
)
def test_dynamic_cv_backend_state_only_qualifies_completed_pass(effective, rc, readable, compile_on_910_95, state,
                                                                status, confirmed):
    values = metadata(effective)
    _update_dynamic_cv_costmodel_report(
        values,
        SimpleNamespace(compile_on_910_95=compile_on_910_95),
        requested=True,
        rc=rc,
        rc_readable=readable,
    )
    report = json.loads(values["auto_simt_scope_report"])
    backend = report["dynamic_cv_pipeline_backend"]
    assert backend["state"] == state
    assert backend["effective_option"] is effective
    assert backend["fallback_rc"] == (rc if rc > 0 else None)
    candidate = report["loop_sensitivity"]["loops"][0]["stage_dependency_elapsed_prediction"]
    assert candidate["status"] == status
    assert candidate["backend_effective_dynamic_cv_confirmed"] is confirmed
    assert candidate["promotion_allowed"] is confirmed
    assert report["selected_route_elapsed_prediction"] == {"predicted_microseconds": 7.0}
