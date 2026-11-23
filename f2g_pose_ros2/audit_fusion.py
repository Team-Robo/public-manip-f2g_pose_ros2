"""Compare ICP enabled/disabled on a saved fusion acquisition; no ROS requests."""
import argparse
import json
from pathlib import Path
import numpy as np
from .fusion import align_and_fuse_shapes


def compare(capture):
    data = np.load(capture, allow_pickle=False)
    report = dict(capture=str(Path(capture).resolve()), coordinates='base_link', physical_accuracy_verified=False)
    for enabled in (False, True):
        metrics = {}
        try:
            fused = align_and_fuse_shapes(data['completed'], data['observed'],
                camera_from_base=data['camera_from_base'] if 'camera_from_base' in data else None,
                camera_matrix=data['camera_matrix'] if 'camera_matrix' in data else None,
                enable_icp_refinement=enabled, diagnostics=metrics)
            metrics.update(status='ok', centroid=fused.mean(axis=0).tolist())
        except ValueError as exc:
            metrics.update(status='rejected', message=str(exc))
        report['icp_enabled' if enabled else 'icp_disabled'] = metrics
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path, help='fusion.npz written by F2G capture_dir')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = compare(args.capture)
    output = args.output or args.capture.with_name('fusion_comparison.json')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
