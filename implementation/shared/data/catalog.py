"""Reference-derived data/catalog. See README.md."""

from pathlib import Path

import pandas as pd

from PIL import Image

class Operations:
    # Reference cells (zero-based): {'tempo': 9, 'ditto': 8, 'hfl': 8, 'flat': 8}
    def _has_trashnet_classes(self, candidate):
        expected = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
        if not candidate.exists() or not candidate.is_dir():
            return False
        subdirs = {p.name.lower() for p in candidate.iterdir() if p.is_dir()}
        return expected.issubset(subdirs)

    # Reference cells (zero-based): {'tempo': 9, 'ditto': 8, 'hfl': 8, 'flat': 8}
    def scan_classification_folder(self, root_dir):
        root_dir = Path(root_dir)
        rows = []
        class_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
        for class_dir in class_dirs:
            label = class_dir.name.strip().lower()
            for file_path in class_dir.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in self.CONFIG["image_extensions"]
                ):
                    try:
                        with Image.open(file_path) as image:
                            width, height = image.size
                        rows.append(
                            {
                                "path": str(file_path),
                                "label": label,
                                "width": width,
                                "height": height,
                            }
                        )
                    except Exception:
                        pass
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No images were found. Expected: root/class_name/image.jpg")
        classes = sorted(df["label"].unique().tolist())
        class_to_id = {name: idx for idx, name in enumerate(classes)}
        df["label_id"] = df["label"].map(class_to_id)
        return (df, classes, class_to_id)
