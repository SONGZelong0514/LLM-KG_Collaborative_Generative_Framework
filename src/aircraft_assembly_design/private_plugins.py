"""Optional loading and invocation of the private MBSE conversion plugins."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from .config import PATHS
from .plans import versioned_artifact_sort_key


class PrivatePluginUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=2)
def load_private_plugin(module_name: str):
    plugin_path = PATHS.private_plugins / f"{module_name}.py"
    if not plugin_path.exists():
        raise PrivatePluginUnavailable(
            f"Private plugin not found: {plugin_path}. Other system functions remain available."
        )

    qualified_name = f"aircraft_assembly_design_private_{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, plugin_path)
    if spec is None or spec.loader is None:
        raise PrivatePluginUnavailable(f"Cannot load private plugin: {plugin_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise PrivatePluginUnavailable(
            f"Private plugin dependency is missing: {exc.name}. Install the optional MBSE dependencies."
        ) from exc
    return module


def generate_mbse_model() -> Path:
    if not PATHS.plans.exists():
        raise FileNotFoundError("No assembly_plan_*.csv file found in ./plans. Please generate the assembly plan first!")
    csv_files = [
        path for path in PATHS.plans.iterdir()
        if path.is_file() and path.name.startswith("assembly_plan_") and path.suffix == ".csv"
    ]
    if not csv_files:
        raise FileNotFoundError("No assembly_plan_*.csv file found in ./plans. Please generate the assembly plan first!")

    latest_csv = sorted(
        csv_files,
        key=lambda path: versioned_artifact_sort_key(path.name, "assembly_plan_", ".csv"),
    )[-1]
    suffix = latest_csv.name[len("assembly_plan_") : -len(".csv")]
    PATHS.mbse.mkdir(parents=True, exist_ok=True)
    fragment = PATHS.mbse / f"owl_out_{suffix}.txt"
    output = PATHS.mbse / f"assembly_plan_MBSE_{suffix}.owl"

    plugin = load_private_plugin("csv2GOPPRRE")
    if not PATHS.gopprre_ontology.exists():
        raise PrivatePluginUnavailable(f"Private ontology template not found: {PATHS.gopprre_ontology}")
    plugin.build(str(latest_csv), str(fragment))
    plugin.merge_fragment(str(PATHS.gopprre_ontology), str(fragment), str(output))
    fragment.unlink(missing_ok=True)
    return output


def generate_simulation_model() -> Path:
    if not PATHS.mbse.exists():
        raise FileNotFoundError("No assembly_plan_MBSE_*.owl file found in ./MBSE. Please generate the MBSE model first!")
    owl_files = [
        path for path in PATHS.mbse.iterdir()
        if path.is_file() and path.name.startswith("assembly_plan_MBSE_") and path.suffix == ".owl"
    ]
    if not owl_files:
        raise FileNotFoundError("No assembly_plan_MBSE_*.owl file found in ./MBSE. Please generate the MBSE model first!")

    latest_owl = sorted(
        owl_files,
        key=lambda path: versioned_artifact_sort_key(path.name, "assembly_plan_MBSE_", ".owl"),
    )[-1]
    PATHS.simulation.mkdir(parents=True, exist_ok=True)
    output = PATHS.simulation / latest_owl.with_suffix(".m").name
    load_private_plugin("GOPPRRE2sim").owl_to_matlab(str(latest_owl), str(output))
    return output

