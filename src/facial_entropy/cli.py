"""`forge-entropy` console entry point: audit | validate | prepare | run | report | all."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from facial_entropy.config import DEFAULT_FEI_ROOT, load_profile
from facial_entropy.util import PRIMARY_SEED, environment_info, git_info, write_json


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", choices=["smoke", "quick", "main"], default="smoke")
    p.add_argument("--seed", type=int, default=PRIMARY_SEED)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--data-root", type=str, default=None, help=f"dataset root (default search includes {DEFAULT_FEI_ROOT})")
    p.add_argument("--dataset", choices=["auto", "manifest", "lfw", "fei"], default="auto")
    p.add_argument("--manifest", type=str, default=None, help="CSV manifest for --dataset manifest")
    p.add_argument("--download", action="store_true", help="opt-in: allow downloading landmark weights / LFW (1 GB cap)")
    p.add_argument("--threads", type=int, default=None, help="RAYON_NUM_THREADS for the Rust core (default from profile: 4)")
    p.add_argument("--landmark-device", type=str, default=None, help="cpu (default) or mps for the FAN landmark model only")
    p.add_argument("--n-subjects", type=int, default=None, help="override the profile's subject target")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="forge-entropy", description="Regional facial entropy experiments on the Hyperplane-Forge surface")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("audit", "inspect Rust/API integration and data availability"),
                        ("validate", "run numerical/control checks (Experiment 0) and emit controls.json"),
                        ("prepare", "discover data, manifest, align, split, calibrate, lock protocol"),
                        ("run", "execute experiments with resumable caches"),
                        ("report", "render figures and results.md from saved results"),
                        ("all", "prepare + validate + run + report")):
        p = sub.add_parser(name, help=help_)
        _common(p)
        if name in ("run", "all"):
            p.add_argument("--experiments", type=str, default=None, help="comma list among exp0,exp1,exp2,exp3,exp4,exp5,exp6 (default: profile)")
    return ap


def _context(args: argparse.Namespace):
    from facial_entropy.experiments import RunContext  # noqa: PLC0415

    cfg = load_profile(args.profile)
    if args.n_subjects is not None:
        cfg.n_subjects = args.n_subjects
    if args.landmark_device:
        cfg.landmark_device = args.landmark_device
    if args.threads is not None:
        cfg.solver.rayon_threads = args.threads
    os.environ.setdefault("RAYON_NUM_THREADS", str(cfg.solver.rayon_threads))
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return RunContext(cfg=cfg, seed=args.seed, out=out, data_root=args.data_root, dataset=args.dataset, manifest_path=args.manifest, download=args.download)


def cmd_audit(args: argparse.Namespace) -> int:
    from facial_entropy.alignment import landmark_asset_status  # noqa: PLC0415
    from facial_entropy.datasets import dataset_summary, discover_fei  # noqa: PLC0415
    from facial_entropy.experiments import FORGE_REPO, source_digest  # noqa: PLC0415
    from facial_entropy.rust_adapter import RustUnavailable, core_info  # noqa: PLC0415

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {"environment": environment_info(), "forge_repo": str(FORGE_REPO), "forge_repo_git": git_info(FORGE_REPO)}
    try:
        rep["core"] = core_info()
        rep["source_digest"] = source_digest()
        rep["rust_ok"] = True
    except RustUnavailable as exc:
        rep["rust_ok"] = False
        rep["rust_error"] = str(exc)
    rep["landmark_assets"] = landmark_asset_status()
    root = Path(args.data_root) if args.data_root else DEFAULT_FEI_ROOT
    recs = discover_fei(root)
    rep["data"] = {"root": str(root), **dataset_summary(recs)}
    write_json(out / "audit.json", rep)
    print(f"Rust extension: {'OK' if rep['rust_ok'] else 'MISSING - ' + rep.get('rust_error', '')}")
    print(f"core: {rep.get('core', {}).get('core_file')}")
    print(f"landmark assets missing: {list(rep['landmark_assets']['missing'])} (download with --download; est. {rep['landmark_assets']['estimated_download_bytes']/1e6:.0f} MB)")
    print(f"data: {rep['data']}")
    print(f"written {out / 'audit.json'}")
    return 0 if rep["rust_ok"] else 2


def cmd_validate(args: argparse.Namespace) -> int:
    from facial_entropy.controls import run_all_controls  # noqa: PLC0415

    ctx = _context(args)
    t = time.perf_counter()
    rgb = None
    if not ctx.cfg.synthetic_only:
        # use one real aligned face for the solver audit when a prepared output exists
        try:
            from facial_entropy.experiments import load_prepared  # noqa: PLC0415

            load_prepared(ctx)
            row = ctx.cohort.primary[ctx.cohort.primary["split"] == "calibration"].iloc[0]
            face = ctx.face_for(row)
            rgb = face.rgb if face is not None else None
        except Exception:  # noqa: BLE001
            rgb = None
    res = run_all_controls(ctx.cfg, ctx.seed, rgb)
    res["runtime_s"] = time.perf_counter() - t
    res["solver_audit_input"] = "real calibration face" if rgb is not None else "synthetic pseudo-face"
    write_json(ctx.out / "controls.json", res)
    s = res["summary"]
    print(f"controls: {s['n_checks']} checks, integrity failures {s['failed_integrity']}, audit findings {s['audit_findings']}")
    return 0 if not s["failed_integrity"] else 1


def _run_experiments(ctx, exps: list[str]) -> None:
    from facial_entropy.experiments import atlas, primary_inference, run_primary_metrics  # noqa: PLC0415

    df = None
    results = None
    if "exp0" in exps:
        from facial_entropy.controls import run_all_controls  # noqa: PLC0415

        t = time.perf_counter()
        rgb = None
        if not ctx.cfg.synthetic_only:
            row = ctx.cohort.primary[ctx.cohort.primary["split"] == "calibration"].iloc[0]
            face = ctx.face_for(row)
            rgb = face.rgb if face is not None else None
        res = run_all_controls(ctx.cfg, ctx.seed, rgb)
        res["runtime_s"] = time.perf_counter() - t
        res["solver_audit_input"] = "real calibration face" if rgb is not None else "synthetic pseudo-face"
        write_json(ctx.out / "controls.json", res)
        ctx.status["exp0"] = f"done ({len(res['summary']['failed_integrity'])} integrity failures, {len(res['summary']['audit_findings'])} audit findings)"
        ctx.save_run_manifest()
    if "exp1" in exps or "exp2" in exps:
        t = time.perf_counter()
        df = run_primary_metrics(ctx)
        results = primary_inference(ctx, df)
        atlas(ctx, df)
        ctx.status["exp1"] = "done (binary endpoints)"
        ctx.status["exp2"] = "done (surface endpoints + maps + atlas)"
        ctx.timings["exp1_exp2"] = time.perf_counter() - t
        ctx.save_run_manifest()
    else:
        import pandas as pd  # noqa: PLC0415

        if (ctx.out / "image_metrics.csv").exists():
            df = pd.read_csv(ctx.out / "image_metrics.csv")
    for name, fn_path in (("exp3", "facial_entropy.ablations:run_ablations"), ("exp4", "facial_entropy.robustness:run_robustness"),
                          ("exp5", "facial_entropy.policies:run_policies"), ("exp6", "facial_entropy.verification:run_verification")):
        if name not in exps:
            continue
        mod, fn = fn_path.split(":")
        func = getattr(__import__(mod, fromlist=[fn]), fn)
        t = time.perf_counter()
        try:
            r = func(ctx, df) if name == "exp3" else func(ctx)
            ctx.status[name] = "done" if not (name == "exp6" and isinstance(r, dict) and not r.get("ran", True)) else f"not run: {r.get('reason')}"
        except Exception as exc:  # noqa: BLE001
            ctx.status[name] = f"failed: {exc!r}"
            traceback.print_exc()
        ctx.timings[name] = time.perf_counter() - t
        ctx.save_run_manifest()


def cmd_prepare(args: argparse.Namespace) -> int:
    from facial_entropy.experiments import prepare  # noqa: PLC0415

    ctx = _context(args)
    prepare(ctx)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from facial_entropy.experiments import load_prepared  # noqa: PLC0415

    ctx = _context(args)
    load_prepared(ctx)
    exps = args.experiments.split(",") if args.experiments else list(ctx.cfg.experiments)
    _run_experiments(ctx, exps)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from facial_entropy.experiments import load_prepared  # noqa: PLC0415
    from facial_entropy.plotting import make_all_figures  # noqa: PLC0415
    from facial_entropy.reporting import build_report  # noqa: PLC0415
    from facial_entropy.util import read_json  # noqa: PLC0415

    ctx = _context(args)
    load_prepared(ctx)
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(ctx.out / "image_metrics.csv") if (ctx.out / "image_metrics.csv").exists() else None
    results = read_json(ctx.out / "primary_results.json") if (ctx.out / "primary_results.json").exists() else None
    figs = make_all_figures(ctx, df, results)
    ctx.status["report"] = f"done ({len(figs)} figures)"
    ctx.save_run_manifest()
    build_report(ctx.out)
    print(f"report written: {ctx.out / 'results.md'} ({len(figs)} figures)")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    from facial_entropy.experiments import prepare  # noqa: PLC0415
    from facial_entropy.plotting import make_all_figures  # noqa: PLC0415
    from facial_entropy.reporting import build_report  # noqa: PLC0415
    from facial_entropy.util import read_json  # noqa: PLC0415

    ctx = _context(args)
    prepare(ctx)
    exps = args.experiments.split(",") if args.experiments else list(ctx.cfg.experiments)
    _run_experiments(ctx, exps)
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(ctx.out / "image_metrics.csv") if (ctx.out / "image_metrics.csv").exists() else None
    results = read_json(ctx.out / "primary_results.json") if (ctx.out / "primary_results.json").exists() else None
    figs = make_all_figures(ctx, df, results)
    ctx.status["report"] = f"done ({len(figs)} figures)"
    ctx.save_run_manifest()
    build_report(ctx.out)
    print(f"done: {ctx.out / 'results.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"audit": cmd_audit, "validate": cmd_validate, "prepare": cmd_prepare, "run": cmd_run, "report": cmd_report, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
