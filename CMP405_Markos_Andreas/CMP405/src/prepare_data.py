"""Prepare MNIST and the 20 external digit samples.

The repository already includes data/mnist_standard.npz for offline reproducibility.
Run this file only if you intentionally remove/rebuild that file.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "external_digits" / "raw"
PROC = DATA / "external_digits" / "processed"

TEMPLATES = {
0:[[(35,15),(25,20),(18,35),(17,60),(25,78),(40,86),(58,84),(72,72),(78,52),(75,32),(65,18),(50,13),(35,15)]],
1:[[(35,30),(50,16),(50,84)],[(38,84),(65,84)]],
2:[[(25,30),(32,18),(50,13),(68,19),(75,33),(70,45),(58,56),(42,68),(25,84),(77,84)]],
3:[[(27,22),(42,14),(60,16),(73,28),(68,42),(55,50)],[(55,50),(70,55),(75,70),(68,82),(50,87),(31,81)]],
4:[[(65,86),(65,15)],[(65,15),(25,62),(80,62)]],
5:[[(75,16),(30,16),(27,48),(48,45),(66,50),(75,62),(72,76),(60,85),(43,86),(27,79)]],
6:[[(70,20),(55,14),(39,19),(28,34),(23,55),(27,75),(40,86),(58,86),(71,76),(73,62),(66,51),(51,47),(35,53),(26,65)]],
7:[[(23,17),(78,17),(62,39),(50,59),(42,84)]],
8:[[(50,50),(35,44),(27,33),(31,20),(44,13),(58,14),(70,23),(70,36),(61,47),(50,50),(36,55),(27,67),(30,80),(43,88),(59,87),(72,76),(72,63),(64,54),(50,50)]],
9:[[(70,51),(60,55),(44,53),(32,44),(28,31),(34,18),(48,12),(63,16),(72,28),(72,50),(68,69),(58,82),(43,87),(31,82)]]
}


def fetch_mnist(force: bool = False) -> None:
    out = DATA / "mnist_standard.npz"
    if out.exists() and not force:
        print(f"MNIST already available: {out}")
        return
    print("Downloading MNIST from OpenML (requires internet)...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    X = np.asarray(X, dtype=np.uint8).reshape(-1, 28, 28)
    y = np.asarray(y, dtype=np.uint8)
    if X.shape != (70000, 28, 28):
        raise RuntimeError(f"Unexpected MNIST shape: {X.shape}")
    np.savez_compressed(out, X_train=X[:60000], y_train=y[:60000], X_test=X[60000:], y_test=y[60000:])
    print(f"Saved {out}")


def create_external_digits() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(405)
    labels, processed = [], []
    for digit in range(10):
        for variant in range(2):
            image = Image.new("L", (140, 140), 255)
            draw = ImageDraw.Draw(image)
            scale = 1.15
            ox = 12 + int(rng.integers(-3, 4)); oy = 10 + int(rng.integers(-3, 4))
            width = int(13 + rng.integers(-1, 4))
            for stroke in TEMPLATES[digit]:
                points = []
                for x, y in stroke:
                    points.append((ox + (x + rng.normal(0, 1.7))*scale, oy + (y + rng.normal(0, 1.7))*scale))
                draw.line(points, fill=10, width=width, joint="curve")
            image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
            angle = (-4 if variant == 0 else 4) + float(rng.normal(0, 1.2))
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
            filename = f"digit_{digit}_{variant+1}.png"
            image.save(RAW / filename)

            arr = 255 - np.asarray(image, dtype=np.uint8)
            ys, xs = np.where(arr > 20)
            crop = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
            h, w = crop.shape; scale2 = 20 / max(h, w)
            nw, nh = max(1, round(w*scale2)), max(1, round(h*scale2))
            resized = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.Resampling.LANCZOS))
            canvas = np.zeros((28, 28), dtype=np.uint8)
            x0, y0 = (28-nw)//2, (28-nh)//2
            canvas[y0:y0+nh, x0:x0+nw] = resized
            Image.fromarray(canvas).save(PROC / filename)
            labels.append(digit); processed.append(canvas)
    np.savez_compressed(DATA / "external_digits" / "external_digits.npz", X=np.stack(processed), y=np.asarray(labels, dtype=np.uint8))
    print("Created 20 external digit samples.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true", help="Re-download/rebuild MNIST even if local NPZ exists")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    fetch_mnist(force=args.force_download)
    create_external_digits()


if __name__ == "__main__":
    main()
