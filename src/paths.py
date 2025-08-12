from pathlib import Path

print(__file__)
root = Path(__file__).parent.parent
data = Path(root / "data")
logs = Path(root / "logs")
chekpoints = Path(root / "checkpoints")
results = Path(root / "results")