r"""Phase 6: where does the adaptive watermark fail, why, and does one cheap fix help?

    python scripts\p6_failure_analysis.py                       # analyse Phase 5 results
    python scripts\p6_failure_analysis.py --try-fixes 40        # + test candidate fixes on 40 VAL images
    python scripts\p6_failure_analysis.py --apply floor0.5      # adopt a fix (then re-run Phase 5)

Honest-methodology rule: the TEST results are only used to find failure patterns.
Candidate fixes are compared on the VAL split, and the test set is evaluated once
more at the end with the chosen fix. Never pick a fix by its test score.

Reads results/p5_raw.csv (+ results/p5x_raw.csv if Phase 5b ran). Outputs in results/:
  p6_where_it_fails.csv   per attack x severity: who fails where (focus vs base config)
  p6_fails_by_texture.csv the same split by image texture
  p6_images.csv           images ranked by how much worse the focus config does
  p6_cases.csv            numbers for the inspected cases (strength of the blocks that went wrong ...)
  p6_cases/*.png          one picture per case: strength map, wrong blocks, vote per bit
  p6_fixes.csv, p6_fix_stats.csv, p6_fix_variants.json     (with --try-fixes)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from watermark import dataset
from watermark.choice import CHOICE, load_choice, map_params
from watermark.config import TEXTURE_CSV
from watermark.failure import STARVED, case_job
from watermark.jobs import eval_variants_job
from watermark.parallel import check_failures, default_workers, pmap
from watermark.sampling import sample
from watermark.stats import compare
from watermark.textures import load_texture_table

# Output directory for error tables, diagnostic logs, and rebuilt case visualizer artifacts
OUT = Path("results")
# Distortions characterized by global distributed perturbations across all 8x8 spatial blocks
NOISE_KINDS = {"jpeg", "blur", "resize", "noise", "smooth", "regen_0.15", "regen_0.30", "vae"}


def who_fails(att, focus, base, by):
    """Per group: success of both configs and the cases only one of them lost.

    Args:
        att (pd.DataFrame): Attack observation records containing success flags.
        focus (str): Primary configuration under failure analysis (e.g., 'adaptive+ecc').
        base (str): Control baseline configuration (e.g., 'uniform+ecc').
        by (str or list[str]): Column name(s) grouping the comparative breakdown.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (Aggregated failure metrics table, pivoted per-trial table).
    """
    # Pivot tabular trials to compare focus and baseline outcomes side-by-side
    wide = att.pivot_table(index=["image", "attack", "severity"], columns="config",
                           values="success", aggfunc="first")[[focus, base]].astype(bool)
    # Re-attach spatial texture class labels
    wide = wide.join(att.drop_duplicates(["image", "attack", "severity"])
                        .set_index(["image", "attack", "severity"])[["texture"]])
    wide = wide.reset_index()
    g = wide.groupby(by)

    # Compute comparative fail rates and isolate discordant trials
    t = pd.DataFrame({
        f"{focus}": g[focus].mean(), f"{base}": g[base].mean(),
        "focus_only_fails": g.apply(lambda d: int((~d[focus] & d[base]).sum()), include_groups=False),
        "base_only_fails": g.apply(lambda d: int((d[focus] & ~d[base]).sum()), include_groups=False),
        "cases": g.size()})
    # Net recovery advantage for focus configuration (negative values denote focus underperforming baseline)
    t["net_for_focus"] = t.base_only_fails - t.focus_only_fails   # < 0: focus config loses here
    return t.sort_values("net_for_focus"), wide


def pick_cases(wide, focus, base, worst, k):
    """Up to k cases where focus failed but base did not, spread over the worst attacks.

    Args:
        wide (pd.DataFrame): Wide-format trial table indicating success across configs.
        focus (str): Name of focus configuration.
        base (str): Name of control baseline.
        worst (pd.DataFrame): Table of attacks ranked by largest net deficit for focus.
        k (int): Maximum number of individual failure cases to sample and inspect.

    Returns:
        pd.DataFrame: Selected subset of failure records for granular case reconstruction.
    """
    # Identify discordant trials where focus failed but baseline survived
    lost = wide[~wide[focus] & wide[base]]
    if lost.empty:
        lost = wide[~wide[focus]]              # nothing lost only by focus: look at shared failures
    order = [a for a, _ in worst.index] if isinstance(worst.index, pd.MultiIndex) else list(worst.index)
    picks, used = [], set()

    # Round-robin selection across worst attack categories to maximize diagnostic diversity
    while len(picks) < k and len(used) < len(lost):
        added = False
        for a in dict.fromkeys(order):
            cand = lost[(lost.attack == a) & ~lost.index.isin(used)]
            if len(cand):
                i = cand.index[0]
                used.add(i)
                picks.append(lost.loc[i])
                added = True
            if len(picks) >= k:
                break
        if not added:
            break
    return pd.DataFrame(picks)


def hypotheses(where, cases, by_tex, focus):
    """Turn the numbers into concrete things to try (printed, not applied).

    Synthesizes empirical observations into actionable hyperparameter adjustments.

    Args:
        where (pd.DataFrame): Attack-level failure breakdown table.
        cases (pd.DataFrame): Rebuilt case diagnostic metrics.
        by_tex (pd.DataFrame): Failure breakdown table partitioned by texture category.
        focus (str): Evaluated configuration name.

    Returns:
        list[str]: Formatted diagnostic bullet tips and remediation suggestions.
    """
    tips = []
    losing = where[where.net_for_focus < 0]
    if losing.empty:
        tips.append(f"{focus} never loses a case to the baseline: no fix needed; report the gains.")
        return tips
    kinds = [a for a, _ in losing.index]
    # Measure fraction of failure conditions belonging to global noise/low-pass attacks
    noise_share = np.mean([k in NOISE_KINDS for k in kinds])

    # Analyze whether starved blocks (assigned very low relative strength) drove errors
    if len(cases) and cases.focus_starved_share_of_wrong.notna().any():
        sw = cases.focus_starved_share_of_wrong.mean()
        sa = cases.focus_starved_share_all.mean()
        if sa == 0:
            tips.append(f"No block in the inspected {focus} cases had less than {STARVED} x the mean "
                        f"strength, so starved blocks are not the cause.")
        else:
            tips.append(f"In {focus} failures, {sw:.0%} of the wrong blocks were 'starved' "
                        f"(strength < {STARVED} x mean); {sa:.0%} of all blocks are.")
        # If starved blocks are disproportionately represented among corrupted blocks
        if sw > sa + 0.1:
            tips.append("-> the map takes too much strength from some blocks. Try a strength floor "
                        "(variant floor0.5) or pulling the map toward uniform (shrink0.5).")

    # Assess broad attack category tendencies
    if noise_share >= 0.6:
        tips.append("-> losses are mostly noise-type edits (blur/resize/smooth/JPEG/regeneration): "
                    "low-pass edits hit every block, so over-concentrating strength hurts. "
                    "Try alpha_half or shrink0.5.")
    else:
        tips.append("-> losses are mostly erasure edits (inpaint/crop): strength was concentrated in "
                    "regions the edit removed. Try beta+0.5 (favour blocks likely to survive) or shrink0.5.")

    # Texture-specific diagnostic breakdowns
    if by_tex is not None and len(by_tex):
        worst_tex = by_tex.net_for_focus.idxmin()
        if by_tex.net_for_focus.min() < 0:
            advice = {"Smooth": "flat images have few strong blocks to shift strength to; try floor0.5.",
                      "Complex": "textured images force many blocks to be skipped (see skipped_blocks in "
                                 "p6_cases.csv); try shrink0.5 so strength is spread more evenly.",
                      "Medium": "no texture-specific cause; follow the attack-type advice above."}
            tips.append(f"-> {worst_tex} images lose the most: {advice.get(worst_tex, '')}")
    return tips


def fix_variants(p):
    """Candidate fixes built around the current settings p (a dict of learned_map kwargs).

    Args:
        p (dict): Current production strength-mapping hyperparameter dictionary.

    Returns:
        dict[str, dict]: Mapping of variant identifiers to modified parameter dictionaries.
    """
    v = {"current": dict(p),
         # Enforce an embedding strength floor (s_min = 0.5) to prevent over-starvation
         "floor0.5": {**p, "s_min": 0.5},
         # Shrink relative strength deviations 50% closer to uniform weighting
         "shrink0.5": {**p, "shrink": 0.5},
         # Joint intervention: moderate shrinkage combined with an explicit strength floor
         "floor0.5_shrink0.7": {**p, "s_min": 0.5, "shrink": 0.7},
         # Halve exponent alpha governing sensitivity to predicted disturbance
         "alpha_half": {**p, "alpha": p["alpha"] / 2},
         # the opposite bet: give LESS strength to blocks edits disturb most. Under a fixed
         # PSNR budget, very noisy blocks may be hopeless, so their strength is better spent
         # on quiet blocks where it actually flips the outcome.
         "alpha_neg": {**p, "alpha": -0.5}}
    # Shift strength toward blocks with higher predicted survival probability
    v["beta+0.5"] = {**p, "beta": p["beta"] + 0.5}
    return v


def try_fixes(n, workers, model):
    """Evaluates candidate fix variants on held-out validation images under equal PSNR.

    Args:
        n (int): Number of validation split images to sample.
        workers (int): Parallel worker process count.
        model (str or Path): Path to the trained survival predictor model.
    """
    # Load baseline mapping configuration and construct candidate parameter sets
    params = map_params()
    variants = fix_variants(params)
    names = sample(dataset.require("val"), n, seed=2)
    jobs = [("val", nm, ["uniform+ecc"], variants, model) for nm in names]
    rows, failed = [], []

    # Execute parallel variant evaluation sweep across validation images
    for (_, nm, *_), r, err in pmap(eval_variants_job, jobs, workers, desc="fixes (val)"):
        if err:
            failed.append((nm, err))
            if len(failed) <= 2:
                print(f"\n! {nm} failed:\n{err}")
        else:
            rows += r
    check_failures(failed, len(names), "val images")

    # Filter out identity conditions and save evaluation records
    res = pd.DataFrame(rows)
    res = res[res.attack != "none"]
    res.to_csv(OUT / "p6_fixes.csv", index=False)

    # Compute high-level recovery rates and severity-wise performance breakdown
    summary = res.groupby("config").agg(success=("success", "mean"), bit_acc=("bit_acc", "mean"))
    by_sev = res.pivot_table(index="config", columns="severity", values="success")

    # Run paired statistical tests against both the current adaptive settings and uniform baseline
    stats = pd.concat([compare(res, [(v, "current") for v in variants if v != "current"]),
                       compare(res, [(v, "uniform+ecc") for v in variants])], ignore_index=True)
    stats.round(5).to_csv(OUT / "p6_fix_stats.csv", index=False)
    (OUT / "p6_fix_variants.json").write_text(json.dumps(variants, indent=1))

    print(f"\nCandidate fixes on {len(names)} VAL images (all at the same PSNR):")
    print(summary.join(by_sev).sort_values("success", ascending=False).round(4).to_string())
    print("\nPaired difference (95% CI over images):")
    print(stats[["A", "B", "diff", "lo", "hi", "p_value", "significant"]].round(4).to_string(index=False))

    # Identify candidate yielding largest empirical gain over current production settings
    vs_cur = stats[stats.B == "current"].set_index("A")
    best = vs_cur["diff"].idxmax() if len(vs_cur) else None
    if best is not None and vs_cur.loc[best, "diff"] > 0:
        sig = "significantly" if vs_cur.loc[best, "significant"] else "but NOT significantly"
        print(f"\nBest fix: {best} ({vs_cur.loc[best, 'diff']:+.3f} vs current, {sig}).")
        print(f"To adopt it: python scripts\\p6_failure_analysis.py --apply {best}")
        if not vs_cur.loc[best, "significant"]:
            print("A non-significant gain is a judgement call: adopting it is fine, but report it as "
                  "'no significant change' and keep the original numbers in the report too.")
    else:
        print("\nNo candidate beats the current settings on val: keep them, and report the failure "
              "analysis as a limitation instead.")


def apply_fix(name):
    """Adopts a validated fix variant by updating the global configuration file.

    Args:
        name (str): Identifier of the fix variant to apply.
    """
    vpath = OUT / "p6_fix_variants.json"
    if not vpath.exists():
        raise SystemExit("run with --try-fixes first")
    variants = json.loads(vpath.read_text())
    if name not in variants:
        raise SystemExit(f"unknown variant {name!r}; choose from {list(variants)}")

    choice = load_choice()
    backup = OUT / "p4_choice_before_p6.json"
    # Back up pre-modification configuration file
    if CHOICE.exists() and not backup.exists():
        shutil.copy(CHOICE, backup)

    # Update active configuration with adopted variant parameters
    choice.update(variants[name])
    choice["phase6_fix"] = name
    CHOICE.write_text(json.dumps(choice, indent=1))
    print(f"{CHOICE} now uses {name}: {variants[name]} (original kept in {backup}).")
    print("Next: python scripts\\p5_ablation.py --fresh   (then Phase 5b: export round 2 + Colab)")


def main():
    """Main execution function for Phase 6 failure analysis and remediation."""
    ap = argparse.ArgumentParser(description="Analyze Phase 5 failure modes and evaluate candidate fixes.")
    ap.add_argument("--focus", default="adaptive+ecc", help="primary configuration under inspection")
    ap.add_argument("--base", default="uniform+ecc", help="control baseline configuration")
    ap.add_argument("--cases", type=int, default=12, help="failure cases to rebuild and picture")
    ap.add_argument("--try-fixes", type=int, default=0, metavar="N", help="evaluate fixes on N val images")
    ap.add_argument("--apply", default=None, metavar="VARIANT", help="adopt a fix from --try-fixes")
    ap.add_argument("--model", default="models/survival.joblib")
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # Route directly to configuration adoption if --apply flag is provided
    if args.apply:
        apply_fix(args.apply)
        return

    # Ingest Phase 5 experimental evaluation log
    raw = OUT / "p5_raw.csv"
    if not raw.exists():
        raise SystemExit("results/p5_raw.csv not found: run Phase 5 (p5_ablation.py) first.")
    run = json.loads((OUT / "p5_run.json").read_text())
    params = {k: run[k] for k in ("alpha", "beta", "shrink", "s_min") if k in run}
    df = pd.read_csv(raw, dtype={"image": str})

    # Associate image identifiers with Laplacian texture classes
    tex = load_texture_table(TEXTURE_CSV)["texture_category"] if TEXTURE_CSV.exists() else pd.Series(dtype=str)
    att = df[df.attack != "none"].copy()
    att["texture"] = att.image.map(tex).fillna("unknown") if len(tex) else "unknown"

    # ---- 1. Where: Attack and Texture Breakdowns ----
    where, wide = who_fails(att, args.focus, args.base, ["attack", "severity"])
    by_tex, _ = who_fails(att, args.focus, args.base, "texture")
    where.to_csv(OUT / "p6_where_it_fails.csv")
    by_tex.to_csv(OUT / "p6_fails_by_texture.csv")   # p6_by_texture.csv belongs to p6_benchmark.py

    print(f"{df.image.nunique()} test images. Cases {args.focus} loses vs {args.base} "
          f"(net < 0 = {args.focus} worse):")
    print(where.head(10).round(3).to_string())
    print("\nWhere it gains the most:")
    print(where.tail(5).iloc[::-1].round(3).to_string())
    print("\nBy texture:")
    print(by_tex.round(3).to_string())

    # Incorporate real Stable Diffusion edit results if Phase 5b has been executed
    p5x = OUT / "p5x_raw.csv"
    if p5x.exists():
        x = pd.read_csv(p5x, dtype={"image": str})
        x = x[x.edit != "none"].assign(attack=lambda d: d.edit, severity="real", texture="-")
        if {args.focus, args.base} <= set(x.config):
            wx, _ = who_fails(x, args.focus, args.base, ["attack", "severity"])
            wx.to_csv(OUT / "p6_where_it_fails_real_edits.csv")
            print("\nReal Stable Diffusion edits (Phase 5b):")
            print(wx.round(3).to_string())

    # ---- 2. Which Images: Identify Most Problematic Host Assets ----
    per_img = (att.groupby(["image", "config"]).success.apply(lambda s: int((~s.astype(bool)).sum()))
                  .unstack()[[args.focus, args.base]].rename(columns=lambda c: f"fails_{c}"))
    per_img["extra_fails_focus"] = per_img[f"fails_{args.focus}"] - per_img[f"fails_{args.base}"]
    per_img["texture"] = per_img.index.map(tex) if len(tex) else "unknown"
    per_img.sort_values("extra_fails_focus", ascending=False).to_csv(OUT / "p6_images.csv")

    # ---- 3. Why: Rebuild Representative Failure Cases ----
    cases = pd.DataFrame()
    if args.cases:
        picks = pick_cases(wide, args.focus, args.base, where, args.cases)
        jobs = [(r.image, r.attack, r.severity, args.focus, args.base, run["model"], params,
                 str(OUT / "p6_cases")) for r in picks.itertuples()]
        rows = []

        # Re-run granular embedding and extraction to log block-level diagnostics and visual plots
        for job, row, err in pmap(case_job, jobs, args.workers, desc="rebuild cases"):
            if err:
                print(f"\n! {job[0]} {job[1]}: {err}")
            else:
                rows.append(row)
        cases = pd.DataFrame(rows)

        if len(cases):
            expect = wide.set_index(["image", "attack", "severity"])
            key = list(zip(cases.image, cases.attack, cases.severity))
            same = (cases.focus_success.values == expect.loc[key, args.focus].values).mean()
            cases.to_csv(OUT / "p6_cases.csv", index=False)
            print(f"\nRebuilt {len(cases)} cases (pictures in results/p6_cases/); "
                  f"{same:.0%} reproduce the Phase 5 result exactly"
                  + ("" if same == 1 else "   <- should be 100%: was the model retrained after Phase 5?"))
            cols = ["focus_wrong_bits", "base_wrong_bits", "focus_wrong_blocks", "base_wrong_blocks",
                    "focus_rel_strength_wrong", "focus_starved_share_of_wrong", "focus_starved_share_all",
                    "focus_skipped_blocks"]
            print(cases[cols].mean().round(3).to_string())

    # ---- 4. What to Try: Generate Empirical Hypotheses ----
    print("\nWhat the numbers suggest:")
    for t in hypotheses(where, cases, by_tex, args.focus):
        print("  " + t)

    # Optionally execute validation sweep across candidate fixes
    if args.try_fixes:
        try_fixes(args.try_fixes, args.workers, args.model)
    else:
        print("\nNext: python scripts\\p6_failure_analysis.py --try-fixes 40   (compares the fixes on VAL)")


if __name__ == "__main__":
    main()
