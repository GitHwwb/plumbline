import argparse
import functools
import glob
import os
import numpy as np
from plumbline import __version__
from plumbline import io as pio
from plumbline import discover
from plumbline.coherence import analyze_tiles
from plumbline.score import flag_tiles, trace_health, input_warning
from plumbline.report import write_report, write_json
from plumbline.dashboard import render_index, thumbnail_b64
from plumbline.model import IndexRow


def _load_inputs(args):
    if args.segment_id:
        ink, mask = pio.fetch_segment(args.segment_id, scroll=args.scroll)
        meta = {"segment_id": args.segment_id, "scroll": args.scroll}
        return ink, mask, meta
    zopts = dict(component=args.zarr_component, z=args.zarr_z, reduce=args.zarr_reduce)
    ink = pio.load_input01(args.ink, **zopts)
    if args.mask:
        mask = pio.load_input01(args.mask, **zopts) > 0.0
    else:
        mask = np.ones(ink.shape, dtype=bool)
    meta = {"segment_id": os.path.splitext(os.path.basename(str(args.ink).rstrip("/")))[0],
            "scroll": args.scroll}
    return ink, mask, meta


def _cmd_run(args) -> int:
    ink, mask, meta = _load_inputs(args)
    feats = analyze_tiles(ink, mask, tile=args.tile, overlap=args.overlap)
    flags = flag_tiles(feats, ink=ink)          # ink enables the seam detector
    report = trace_health(feats, flags)
    write_report(args.output, meta, ink, feats, flags, report)
    if args.json:
        write_json(args.json, meta, feats, flags, report,
                   params={"overlap": args.overlap})
    print(f"{meta['segment_id']}: trace health {report.score}/100 "
          f"(orient {report.n_orient}, spacing {report.n_spacing}, "
          f"garble {report.n_garble}, seam {report.n_seam}) -> {args.output}")
    warn = input_warning(feats, flags)
    if warn:
        print(f"  warning: {warn}")
    return 0


def _eval_segment(s, scroll, tile, overlap, out_dir, no_reports, no_thumbnails):
    """Evaluate ONE segment into its IndexRow (and write its report file).
    Module-level (not a closure) so ProcessPoolExecutor can pickle it for
    --jobs; exceptions are converted to an error row INSIDE the worker so one
    bad segment can never abort the batch, parallel or not."""
    try:
        ink = pio.load_input01(s.ink_path)   # image file OR Zarr store
        mask = pio.load_mask(s.mask_path) if s.mask_path else np.ones(ink.shape, dtype=bool)
        feats = analyze_tiles(ink, mask, tile=tile, overlap=overlap)
        flags = flag_tiles(feats, ink=ink)          # ink enables the seam detector
        rep = trace_health(feats, flags)
        report_filename = None
        if not no_reports:
            report_filename = f"{s.seg_id}.html"
            write_report(os.path.join(out_dir, report_filename),
                         {"segment_id": s.seg_id, "scroll": scroll},
                         ink, feats, flags, rep)
        thumb = None if no_thumbnails else thumbnail_b64(ink, feats, flags)
        return IndexRow(s.seg_id, rep.score, rep.n_orient, rep.n_spacing,
                        rep.n_garble, rep.low_conf_frac,
                        report_filename, thumb, None, n_seam=rep.n_seam)
    except Exception as e:  # one bad segment must not abort the whole batch
        return IndexRow(s.seg_id, 0, 0, 0, 0, 1.0, None, None, str(e))


def _cmd_batch(args) -> int:
    segs = discover.find_segments(args.segments_dir)
    n_subdirs = sum(1 for p in glob.glob(os.path.join(args.segments_dir, "*"))
                    if os.path.isdir(p))
    skipped = n_subdirs - len(segs)
    os.makedirs(args.output, exist_ok=True)

    worker = functools.partial(_eval_segment, scroll=args.scroll, tile=args.tile,
                               overlap=args.overlap, out_dir=args.output,
                               no_reports=args.no_reports,
                               no_thumbnails=args.no_thumbnails)
    if args.jobs > 1:
        # Segments are independent; executor.map preserves input order, so the
        # dashboard is identical to a sequential run -- a throughput knob only.
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(worker, segs))
    else:
        rows = [worker(s) for s in segs]
    n_err = sum(1 for r in rows if r.error is not None)

    index_path = os.path.join(args.output, "index.html")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(render_index(rows, {"scroll": args.scroll}))
    print(f"processed {len(rows) - n_err}, skipped {skipped}, errored {n_err} "
          f"-> {index_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plumbline",
                                description="Trace-quality reports for scroll segments")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="analyze a segment ink prediction")
    run.add_argument("ink", nargs="?", help="path to ink-prediction PNG/TIF or Zarr store")
    run.add_argument("--segment-id", help="fetch by segment id (needs [fetch] extra)")
    run.add_argument("--scroll", default="Scroll1")
    run.add_argument("--mask", help="path to flat_mask image")
    run.add_argument("--zarr-component",
                     help="array name / multiscale level within a Zarr group "
                          "(default: highest-resolution level)")
    run.add_argument("--zarr-z", type=int, default=None,
                     help="for a 3-D Zarr, the plane index to analyze")
    run.add_argument("--zarr-reduce", choices=["max", "mean"], default=None,
                     help="for a 3-D Zarr, project the first axis instead of "
                          "picking one plane")
    run.add_argument("-o", "--output", default="report.html")
    run.add_argument("--json", help="also write a JSON sidecar to this path")
    run.add_argument("--tile", type=int, default=None, help="tile size in px (default: auto-adapt to text scale)")
    run.add_argument("--overlap", type=float, default=0.5)
    run.set_defaults(func=_cmd_run)
    batch = sub.add_parser("batch", help="analyze a folder of segments into a dashboard")
    batch.add_argument("segments_dir", help="folder whose subfolders are segments")
    batch.add_argument("-o", "--output", default="plumbline_out",
                       help="output dir for index.html + per-segment reports")
    batch.add_argument("--scroll", default="Scroll1")
    batch.add_argument("--tile", type=int, default=None, help="tile size in px (default: auto-adapt to text scale)")
    batch.add_argument("--overlap", type=float, default=0.5)
    batch.add_argument("--jobs", type=int, default=1,
                       help="parallel worker processes (segments are independent; "
                            "results identical to a sequential run)")
    batch.add_argument("--no-reports", action="store_true",
                       help="skip per-segment reports (dashboard only)")
    batch.add_argument("--no-thumbnails", action="store_true",
                       help="skip row thumbnails")
    batch.set_defaults(func=_cmd_batch)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run" and not args.ink and not args.segment_id:
        raise SystemExit("provide an ink path or --segment-id")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
