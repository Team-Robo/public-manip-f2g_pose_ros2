"""Exercise the ROS executable without ROS networking or package metadata lookup."""
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import importlib.metadata
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('installed', [False, True])
def test_launcher_directly_imports_main_even_with_stale_distribution_metadata(monkeypatch, installed):
    executable=ROOT/'scripts/object_perception_node'
    if installed:
        executable=ROOT.parents[1]/'install/f2g_pose_ros2/lib/f2g_pose_ros2/object_perception_node'
        if not executable.exists():
            pytest.skip('Package not installed in this workspace')
    main=Mock()
    monkeypatch.setitem(sys.modules,'f2g_pose_ros2.object_perception_node',SimpleNamespace(main=main))
    metadata=Mock(side_effect=AssertionError('Launcher must not query stale console entry points'))
    monkeypatch.setattr(importlib.metadata,'distribution',metadata)
    runpy.run_path(str(executable),run_name='__main__')
    main.assert_called_once_with()
    metadata.assert_not_called()


def test_setup_installs_direct_launcher_without_a_conflicting_console_entry(monkeypatch):
    import setuptools
    setup=Mock()
    monkeypatch.setattr(setuptools,'setup',setup)
    runpy.run_path(str(ROOT/'setup.py'))
    config=setup.call_args.kwargs
    assert 'scripts/object_perception_node' in config['scripts']
    assert not any(s.startswith('object_perception_node =') for s in config['entry_points']['console_scripts'])
