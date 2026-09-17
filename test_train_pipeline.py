from pathlib import Path
import shutil

from src import train


def test_main_creates_processed_directory_if_missing():
    processed_dir = Path(train.ROOT) / "data" / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    train.main()

    assert processed_dir.exists()
    assert (processed_dir / "model_features.csv").exists()

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
