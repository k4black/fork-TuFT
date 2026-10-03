"""Keep docs/compatibility.md in sync with the installed tinker SDK and the server routes."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from tinker.lib.public_interfaces.rest_client import RestClient
from tinker.lib.public_interfaces.sampling_client import SamplingClient
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker.lib.public_interfaces.training_client import TrainingClient

from tuft.config import AppConfig, ModelConfig
from tuft.server import create_root_app


TABLE = Path(__file__).parents[1] / "docs" / "compatibility.md"
ROW = re.compile(r"^\| `(\w+\.\w+)` \| ([\w-]+) \| ([^|]+) \|", re.MULTILINE)


def _norm(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def test_compat_matrix(tmp_path: Path) -> None:
    rows = {name: (status, route.strip()) for name, status, route in ROW.findall(TABLE.read_text())}

    sdk = {
        f"{cls.__name__}.{name.removesuffix('_async')}"
        for cls in (ServiceClient, TrainingClient, SamplingClient, RestClient)
        for name, _ in inspect.getmembers(cls, callable)
        if not name.startswith("_")
    }
    missing = sorted(sdk - rows.keys())
    assert not missing, f"Add rows to docs/compatibility.md for: {missing}"

    config = AppConfig(
        checkpoint_dir=tmp_path,
        supported_models=[ModelConfig(model_name="m", model_path=Path("/m"), max_model_len=8)],
    )
    registered = {
        f"{method} {_norm(route.path)}"
        for route in create_root_app(config).routes
        for method in getattr(route, "methods", None) or ()
    }
    wrong = []
    for name, (status, route) in rows.items():
        if status in ("supported", "partial") and _norm(route) not in registered:
            wrong.append(f"{name}: {status} but {route} is not registered")
        elif status == "unsupported" and _norm(route) in registered:
            wrong.append(f"{name}: unsupported but {route} is registered")
        elif status not in ("supported", "partial", "unsupported", "client-side"):
            wrong.append(f"{name}: unknown status {status}")
    assert not wrong, "\n".join(wrong)
