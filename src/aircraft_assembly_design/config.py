"""Central configuration and filesystem locations for the application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parent
REPOSITORY_ROOT = SOURCE_ROOT.parent
load_dotenv(Path.cwd() / ".env")


def _runtime_root() -> Path:
    configured = os.getenv("AIRCRAFT_DESIGN_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    if (Path.cwd() / ".env").exists() or (Path.cwd() / "pyproject.toml").exists():
        return Path.cwd().resolve()
    return REPOSITORY_ROOT


PROJECT_ROOT = _runtime_root()
if PROJECT_ROOT != Path.cwd().resolve():
    load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AppPaths:
    root: Path = PROJECT_ROOT

    @property
    def assets(self) -> Path:
        return PACKAGE_DIR / "assets"

    @property
    def plans(self) -> Path:
        return self.root / "plans"

    @property
    def constraints(self) -> Path:
        return self.root / "constraints"

    @property
    def mbse(self) -> Path:
        return self.root / "MBSE"

    @property
    def simulation(self) -> Path:
        return self.root / "Simulation"

    @property
    def verification(self) -> Path:
        return self.root / "Verification"

    @property
    def static(self) -> Path:
        return self.root / "static"

    @property
    def private_plugins(self) -> Path:
        configured = os.getenv("AIRCRAFT_DESIGN_PRIVATE_PLUGINS_DIR")
        return Path(configured).expanduser().resolve() if configured else self.root / "plugins" / "private"

    @property
    def logo(self) -> Path:
        return self.assets / "logo.png"

    @property
    def gopprre_ontology(self) -> Path:
        return self.private_plugins / "GOPPRRE.owl"


@dataclass(frozen=True)
class AppSettings:
    model: str = os.getenv("AIRCRAFT_DESIGN_MODEL", "gpt-5.6-sol")
    temperature: float = float(os.getenv("AIRCRAFT_DESIGN_TEMPERATURE", "0"))
    host: str = os.getenv("AIRCRAFT_DESIGN_HOST", "localhost")
    port: int = int(os.getenv("AIRCRAFT_DESIGN_PORT", "7860"))
    gpu_count: int = int(os.getenv("AIRCRAFT_DESIGN_GPU_COUNT", "4"))
    gpu_poll_interval: float = float(os.getenv("AIRCRAFT_DESIGN_GPU_POLL_INTERVAL", "0.25"))


PATHS = AppPaths()
SETTINGS = AppSettings()

