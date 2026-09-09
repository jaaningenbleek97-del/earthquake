from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def data_dir(root: Path | None = None) -> Path:
    base = root or repo_root()
    path = base / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(root: Path | None = None) -> Path:
    path = data_dir(root) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_dir(root: Path | None = None) -> Path:
    path = data_dir(root) / "features"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prediction_dir(root: Path | None = None) -> Path:
    path = data_dir(root) / "predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prediction_batches(root: Path | None = None, dest: Path | None = None) -> list[Path]:
    """Card files only. Sidecars are `batch-*.meta.json` and must not load as cards."""
    folder = dest if dest is not None else prediction_dir(root)
    return [
        path
        for path in sorted(folder.glob("batch-*.json"))
        if not path.name.endswith(".meta.json")
    ]


def model_dir(root: Path | None = None) -> Path:
    path = data_dir(root) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
