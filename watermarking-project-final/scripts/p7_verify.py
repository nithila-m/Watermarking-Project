r"""Phase 7: verify every reported number against the raw logs, then write the report tables.

    python scripts\p7_verify.py

Checks (each PASS / WARN / FAIL, saved to results/p7_checks.csv):
  settings   Phase 5 ran with the CURRENT model file and map settings (not stale)
  complete   every test image has every config x attack, no duplicates
  psnr       every config is calibrated to the target PSNR (fair comparison)
  clean      with no edit, the ECC configs recover 100% of messages
  tables     p5_by_severity / p5_by_texture / p5_stats recompute exactly from p5_raw.csv
  texture    the test sample is balanced across Smooth / Medium / Complex
  phase4     the map was chosen on the VAL split (not on test)
  real edits Phase 5b used the same model + settings, all files present, clean row = 100%
Then writes results/final_numbers.md: the tables and significance statements for the
report, each labelled with the file it came from. Re-run after any change; if any
check FAILs, fix that before copying numbers into the report.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from watermark.choice import file_md5, load_choice, map_params
from watermark.config import TARGET_PSNR, TEXTURE_CSV
from watermark.external import DEFAULT_EXPORT, edit_names
from watermark.stats import compare
from watermark.textures import load_texture_table

# Output directory for verification audit logs and final summary tables
OUT = Path("results")
# Canonical evaluation ordering for configurations
ORDER = ["uniform", "uniform+ecc", "heuristic+ecc", "adaptive", "adaptive+ecc"]
# Numerical tolerance threshold for rounding differences in saved tables
TOL = 6e-4          # saved tables are rounded to 4 decimals
checks = []


def check(name, ok, detail, warn_only=False):
    """Logs an audit check outcome, printing the status and recording details.

    Args:
        name (str): Concise identifier for the audit check.
        ok (bool): Boolean assertion representing pass/fail state.
        detail (str): Descriptive diagnostic details or failure reason.
        warn_only (bool): If True, a failed assertion results in a WARN rather than a FAIL.

    Returns:
        bool: True if the assertion passed, False otherwise.
    """
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    checks.append({"check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")
    return ok


def md_table(df, pct=True, digits=1):
    """Plain markdown table (no extra packages). Fractions shown as percentages.

    Args:
        df (pd.DataFrame): Tabular data to render.
        pct (bool): Whether to format fractional floating values as percentages.
        digits (int): Decimal precision for percentage values.

    Returns:
        str: Formatted Markdown table string.
    """
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for c, v in row.items():
            # Format performance rates as percentages, excluding absolute image fidelity metrics
            if isinstance(v, (float, np.floating)) and pct and str(c) not in ("psnr", "ssim", "PSNR (dB)", "SSIM"):
                cells.append("-" if np.isnan(v) else f"{100 * v:.{digits}f}%")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4f}" if str(c) in ("ssim", "SSIM") else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def ci_sentence(r):
    """Formats bootstrap confidence intervals and McNemar significance into a natural sentence.

    Args:
        r (pd.Series or dict): Record containing pairwise statistical comparison results.

    Returns:
        str: Human-readable sentence summarizing difference, confidence intervals, and p-values.
    """
    sig = "significant" if r["significant"] else "not significant"
    return (f"{r['A']} vs {r['B']}: {100 * r['diff']:+.1f} points "
            f"[95% CI {100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}], McNemar p = {r['p_value']:.3g} ({sig}; "
            f"{r['n_images']} images, {r['n_cases']} cases)")


def compare_saved(name, recomputed, path):
    """Compares dynamically recomputed DataFrame against a previously saved CSV file.

    Args:
        name (str): Audit check title.
        recomputed (pd.DataFrame): In-memory re-aggregated DataFrame from raw logs.
        path (Path): Path to the static saved CSV on disk.

    Returns:
        bool: True if files match within numerical tolerance TOL, False otherwise.
    """
    if not path.exists():
        return check(name, False, f"{path} missing (re-run the phase that writes it)")
    saved = pd.read_csv(path, index_col=0)
    common_r = [i for i in recomputed.index if i in saved.index]
    common_c = [c for c in recomputed.columns if c in saved.columns]
    # Calculate maximum absolute delta across all overlapping row-column cells
    diff = (recomputed.loc[common_r, common_c] - saved.loc[common_r, common_c]).abs().max().max()
    return check(name, diff <= TOL and len(common_r) == len(recomputed),
                 f"max difference {diff:.2g} between {path.name} and p5_raw.csv")


def main():
    """Main verification routine auditing experimental integrity and generating report numbers."""
    ap = argparse.ArgumentParser(description="Verify evaluation logs and generate report tables.")
    ap.add_argument("--model", default="models/survival.joblib", help="path to trained model")
    ap.add_argument("--external", default="data/external", help="path to external edited images")
    ap.add_argument("--export", default=str(DEFAULT_EXPORT), help="path to exported baseline watermarks")
    args = ap.parse_args()

    raw_p = OUT / "p5_raw.csv"
    if not raw_p.exists():
        raise SystemExit("results/p5_raw.csv not found: run Phase 5 first.")
    df = pd.read_csv(raw_p, dtype={"image": str})
    run = json.loads((OUT / "p5_run.json").read_text()) if (OUT / "p5_run.json").exists() else {}
    params = map_params()
    md5 = file_md5(args.model)

    # ---- 1. Verify Experimental Settings and Model Provenance ----
    same_params = all(abs(run.get(k, -1) - v) < 1e-9 for k, v in params.items())
    check("settings: map", same_params, f"p5 ran with { {k: run.get(k) for k in params} }, "
                                        f"current choice {params}")
    if "model_md5" in run:
        check("settings: model", run["model_md5"] == md5,
              f"model used by p5 {run['model_md5']}, model on disk {md5}"
              + ("" if run["model_md5"] == md5 else " -> retrained after Phase 5: re-run p5 --fresh"))
    else:
        check("settings: model", False, "p5_run.json has no model fingerprint (made by an older p5); "
                                        "re-run p5_ablation.py --fresh to be sure", warn_only=True)

    # ---- 2. Dataset Completeness and Row Alignment ----
    configs = sorted(set(df.config))
    n_att = df[df.config == configs[0]].groupby("image").size()
    per_img = df.groupby("image").size()
    dups = int(df.duplicated(["image", "config", "attack", "severity"]).sum())
    check("complete", set(configs) == set(ORDER) and per_img.nunique() == 1 and dups == 0,
          f"{df.image.nunique()} images x {len(configs)} configs x {int(n_att.iloc[0])} attack rows; "
          f"{dups} duplicates; rows per image {sorted(int(v) for v in per_img.unique())}")

    # ---- 3. PSNR Transparency Calibration and Clean Extraction ----
    clean = df[df.attack == "none"]
    q = clean.groupby("config")[["psnr", "ssim", "success"]].mean().reindex(ORDER)
    low = int((clean.psnr < TARGET_PSNR - 0.5).sum())
    check("psnr", (q.psnr - TARGET_PSNR).abs().max() < 0.15 and low == 0,
          f"mean PSNR per config {q.psnr.round(2).to_dict()}; {low} image/config pairs below "
          f"{TARGET_PSNR - 0.5} dB", warn_only=low > 0 and (q.psnr - TARGET_PSNR).abs().max() < 0.15)
    ecc_clean = q.loc[[c for c in ORDER if c.endswith("ecc")], "success"]
    check("clean", (ecc_clean == 1).all(), f"no-edit success for ECC configs {ecc_clean.round(4).to_dict()}",
          warn_only=(ecc_clean > 0.99).all())

    # ---- 4. Recompute and Verify Aggregate Tables ----
    tex = load_texture_table(TEXTURE_CSV)["texture_category"] if TEXTURE_CSV.exists() else pd.Series(dtype=str)
    att = df[df.attack != "none"].copy()
    att["texture"] = att.image.map(tex).fillna("unknown") if len(tex) else "unknown"

    # Pivot recomputation for severity-wise extraction survival
    by_sev = att.pivot_table(index="config", columns="severity", values="success").reindex(ORDER)
    by_sev["all attacks"] = att.groupby("config").success.mean()
    by_sev["bit_acc"] = att.groupby("config").bit_acc.mean()

    # Pivot recomputation across Smooth, Medium, and Complex texture tiers
    by_tex = att.pivot_table(index="config", columns="texture", values="success").reindex(ORDER)
    by_tex = by_tex[[c for c in ("Smooth", "Medium", "Complex", "unknown") if c in by_tex]]

    compare_saved("tables: severity", by_sev, OUT / "p5_by_severity.csv")
    compare_saved("tables: texture", by_tex, OUT / "p5_by_texture.csv")

    # Pairwise statistical comparisons
    pairs = [("adaptive+ecc", "uniform+ecc"), ("heuristic+ecc", "uniform+ecc"),
             ("adaptive", "uniform"), ("uniform+ecc", "uniform")]
    stats_all = compare(att, pairs)
    stats_sev = compare(att, pairs, by="severity")
    sp = OUT / "p5_stats.csv"
    if sp.exists():
        saved = pd.read_csv(sp)
        saved = saved[saved.split_by == "all"].set_index(["A", "B"])
        mine = stats_all.set_index(["A", "B"])
        d = (mine[["diff", "lo", "hi"]] - saved.loc[mine.index, ["diff", "lo", "hi"]]).abs().max().max()
        check("tables: stats", d <= 1e-4, f"max difference {d:.2g} between p5_stats.csv and a fresh bootstrap")
    else:
        check("tables: stats", False, "p5_stats.csv missing")

    # ---- 5. Test Split Texture Balance Audit ----
    tc = att.drop_duplicates("image").texture.value_counts()
    check("texture", tc.max() - tc.min() <= max(2, 0.05 * tc.sum()), f"test images per texture {tc.to_dict()}",
          warn_only=True)

    # ---- 6. Phase 4 Validation Hygiene (Preventing Test Leakage) ----
    choice = load_choice()
    check("phase4", choice.get("val_images", 0) > 0,
          f"map chosen on {choice.get('val_images', 0)} VAL images: {choice.get('variant', '?')}"
          + (f", Phase 6 fix: {choice['phase6_fix']}" if "phase6_fix" in choice else ""))

    # ---- 7. External Generative AI Edits Audit (Phase 5b) ----
    real = None
    xp = OUT / "p5x_raw.csv"
    if xp.exists():
        x = pd.read_csv(xp, dtype={"image": str})
        man_p = Path(args.export) / "manifest.csv"
        if man_p.exists():
            man = pd.read_csv(man_p, dtype={"image": str})
            man = man[(man.split == "test") & man.config.str.contains("adaptive")]
            if len(man) and "model_md5" in man:
                same = (man.model_md5.astype(str) == str(md5)).all() and all(
                    np.allclose(man[k].astype(float), v) for k, v in params.items() if k in man)
                check("real edits: settings", same,
                      "exported adaptive images used the current model and map settings" if same else
                      "exported adaptive images used a different model/settings -> re-export round 2 "
                      "and re-run Colab, or report 5b as made with the earlier settings")
            else:
                check("real edits: settings", False, "manifest has no model fingerprint (older export)",
                      warn_only=True)
        none_ok = x[x.edit == "none"].success.mean()
        check("real edits: clean", none_ok == 1, f"exported images before editing recover {none_ok:.1%}")
        edits = [e for e in edit_names(args.external)]
        combos = x[x.edit != "none"].groupby(["edit", "config"]).image.nunique()
        check("real edits: complete", combos.nunique() == 1,
              f"{len(edits)} edits; images per edit x config {sorted(int(v) for v in combos.unique())}", warn_only=True)
        real = x[x.edit != "none"].pivot_table(index="edit", columns="config", values="success")
        real = real[[c for c in ORDER if c in real]]
        sx = OUT / "p5x_by_edit.csv"
        if sx.exists():
            saved = pd.read_csv(sx, index_col=0)
            d = (real - saved.loc[real.index, real.columns]).abs().max().max()
            check("real edits: table", d <= TOL, f"max difference {d:.2g} vs p5x_by_edit.csv")
    else:
        check("real edits", False, "results/p5x_raw.csv not found (Phase 5b not run yet)", warn_only=True)

    # Save validation audit outcomes to CSV
    pd.DataFrame(checks).to_csv(OUT / "p7_checks.csv", index=False)
    n_fail = sum(c["status"] == "FAIL" for c in checks)

    # ---- 8. Construct Final Report Markdown Artifact ----
    lines = [f"# Final numbers (generated {date.today()} by scripts/p7_verify.py)", "",
             f"Checks: {len(checks) - n_fail} passed or warned, {n_fail} failed (details in results/p7_checks.csv)."
             + ("  **Fix the failures before using these numbers.**" if n_fail else ""), "",
             "## Setup", "",
             f"- Test images: {df.image.nunique()} ({', '.join(f'{k} {v}' for k, v in tc.items())}), "
             f"every config calibrated to {TARGET_PSNR} dB PSNR",
             f"- Learned map settings: {params}; model fingerprint {md5}",
             "- Success = the 32-bit message decodes exactly (ECC configs) / all 64 bits right (no ECC)", "",
             "## Table 1. Image quality and clean extraction  (source: results/p5_raw.csv)", "",
             md_table(q.rename(columns={"psnr": "PSNR (dB)", "ssim": "SSIM", "success": "no-edit success"})), "",
             "## Table 2. Success rate by edit severity  (source: results/p5_by_severity.csv)", "",
             md_table(by_sev.rename(columns={"bit_acc": "bit accuracy"})), "",
             "## Table 3. Success rate by image texture  (source: results/p5_by_texture.csv)", "",
             md_table(by_tex), "",
             "## Significance  (source: results/p5_stats.csv)", ""]
    lines += [f"- {ci_sentence(r)}" for _, r in stats_all.iterrows()]
    sev_rows = stats_sev[stats_sev.A == "adaptive+ecc"]
    lines += [f"- severity {r['group']}: {ci_sentence(r)}" for _, r in sev_rows.iterrows()]

    # Append real generative edit tables if available
    if real is not None:
        lines += ["", "## Table 4. Real Stable Diffusion edits  (source: results/p5x_by_edit.csv)", "",
                  md_table(real)]
        case = x[x.edit != "none"].assign(attack=lambda d: d.edit, severity="real")
        if {"adaptive+ecc", "uniform+ecc"} <= set(case.config):
            lines += ["", f"- {ci_sentence(compare(case, [('adaptive+ecc', 'uniform+ecc')]).iloc[0])}"]

    # Evaluate against the explicit 80% project robustness target
    target = by_sev.loc["adaptive+ecc", [c for c in ("light", "moderate", "severe") if c in by_sev]]
    lines += ["", "## 80% target (adaptive+ecc)", "",
              "- Met for: " + (", ".join(f"{k} edits ({v:.1%})" for k, v in target.items() if v >= 0.8) or "none"),
              "- Not met for: " + (", ".join(f"{k} edits ({v:.1%})" for k, v in target.items() if v < 0.8) or "none")]

    if real is not None and "adaptive+ecc" in real:
        r = real["adaptive+ecc"]
        lines += ["- Real AI edits met: " + (", ".join(f"{k} ({v:.1%})" for k, v in r.items() if v >= 0.8) or "none"),
                  "- Real AI edits not met: " + (", ".join(f"{k} ({v:.1%})" for k, v in r.items() if v < 0.8) or "none")]

    (OUT / "final_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{'ALL CHECKS PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}. "
          f"Report tables: results/final_numbers.md")


if __name__ == "__main__":
    main()
