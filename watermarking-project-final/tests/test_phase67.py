"""Phase 6/7 helpers: map knobs, the settings file, and exact case reproduction."""
import json

import numpy as np

from watermark import choice, failure
from watermark.evaluate import CONFIGS, evaluate_image
from watermark.strength_map import learned_map, normalize


class FakeModel:
    """Stands in for the trained forests: disturbance grows across the image."""
    def predict_sigma(self, feats):
        return np.linspace(1.0, 20.0, len(feats))

    def predict_keep(self, feats):
        return np.linspace(0.2, 1.0, len(feats))


def test_learned_map_defaults_unchanged(image):
    m = learned_map(image, FakeModel(), alpha=1.0)
    assert np.allclose(m, normalize(np.linspace(1.0, 20.0, 1024)))


def test_floor_and_shrink(image):
    base = learned_map(image, FakeModel())
    floored = learned_map(image, FakeModel(), s_min=0.5)
    shrunk = learned_map(image, FakeModel(), shrink=0.5)
    assert floored.min() > 0.49 > base.min()     # normalize() rescales after clipping, so ~0.5
    assert shrunk.std() < base.std()
    for m in (floored, shrunk):
        assert abs(m.mean() - 1) < 1e-9


def test_map_params_defaults_and_overrides(tmp_path):
    p = tmp_path / "choice.json"
    assert choice.map_params(p) == choice.MAP_KEYS
    p.write_text(json.dumps({"alpha": 0.5, "s_min": 0.5, "verdict": "x"}))
    got = choice.map_params(p, alpha=None, beta=1.0)
    assert got == {"alpha": 0.5, "beta": 1.0, "shrink": 1.0, "s_min": 0.5}


def test_reproduce_matches_evaluate(image, monkeypatch):
    """A Phase 6 rebuild must give the same answer Phase 5 recorded."""
    monkeypatch.setattr(failure.dataset, "load", lambda split, name: image)
    rows = evaluate_image(image, "astro", {"uniform+ecc": CONFIGS["uniform+ecc"]},
                          attacks=[("smooth", "severe"), ("jpeg", "moderate")])
    for r in rows:
        if r["attack"] == "none":
            continue
        got = failure.reproduce("test", "astro", "uniform+ecc", None, {}, r["attack"], r["severity"])
        assert got["success"] == r["success"]
        assert int(got["wrong_bit"].sum()) == round(r["ber"] * 64)
