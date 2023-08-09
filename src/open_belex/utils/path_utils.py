r"""
 By Dylon Edwards

 Copyright 2023 GSI Technology, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the “Software”), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import inspect
import os
import sys
from pathlib import Path
from typing import Optional


def project_root(project_name: Optional[str] = None) -> Path:
    """Determine the project root by finding a directory
    that contains the hidden file named .project-root.
    The file need not contain any useful information."""

    for frame_info in inspect.stack():
        if frame_info.filename != __file__:
            path = Path(frame_info.filename)
            break

    path = path.parent
    while path != path.parent:
        project_root_file = path / ".project-root"
        if project_root_file.exists():
            if project_name is None:
                return path
            with open(project_root_file, "rt") as f:
                if project_name == f.read().strip():
                    return path
        path = path.parent

    return path


def absolute_gsi_system_root(version_number='120.11.300.7-rc',
                             windows_drive_letter_and_colon: str = '',
                             release: bool = True):
    """Return a prefix for locating GSI system files, often at absolute
    path /efs/data/products/system/<version_number>/<release-or-debug>
    /system-package/gsi-gal-device-lib.

    TODO: this path should be read from a config file, command-line argument
     or environment variable. It's bad to put absolute data in code like this."""

    # TODO: Not tested on Windows!

    root = Path(windows_drive_letter_and_colon + '/')

    result = root / 'efs' / 'data' / 'products' / 'system' / \
        version_number / ('release' if release else 'debug') / \
        'system-package' / 'gsi-gal-device-lib'
    return result


def path_wrt_root(file_path, project_name: Optional[str] = None):
    root_file_path = Path(project_root(project_name), file_path)

    if not root_file_path.exists():
        this_dir = os.path.split(os.path.abspath(__file__))[0]
        prefix = this_dir + "/../../../../../"
        next_file_path = Path(prefix) / "share" / "open-belex" / file_path
        if next_file_path.exists():
            root_file_path = next_file_path

    return root_file_path


def _copy_file(source_path, target_path) -> Path:
    with open(source_path, "rb") as f:
        body = f.read()
    with open(target_path, "wb") as f:
        f.write(body)
    return target_path


def copy_file_wrt_root(file_path_wrt_root: Path, target_path: Path) -> Path:
    """Copy a file from a path relative to the project root to the
    given destination."""
    source_path = path_wrt_root(file_path_wrt_root)
    return _copy_file(source_path, target_path)


def _copy_tree(source, directory):
    directory = Path(directory)
    target = Path(directory, source.name) if directory.exists() else directory
    if source.is_dir():
        if target.exists() and not target.is_dir():
            raise RuntimeError("Source ["+source+"] and target ["+target+"] have conflicting types")

        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_tree(child, target)
    else:
        if target.exists():
            target.unlink()
        _copy_file(source, target)


def copy_tree_wrt_root(dir_path_wrt_root, destination):

    """Copy a directory tree from a path relative to the project root
    to the given destination."""

    dir_path = path_wrt_root(dir_path_wrt_root)
    return _copy_tree(dir_path, destination)


def exists_wrt_root(file_path_wrt_root, project_name: Optional[str] = None):

    """Determine whether a file exists at the given path relative to
    the project root."""

    file_path = path_wrt_root(file_path_wrt_root, project_name)
    return file_path.exists()


def user_tmp():
    if sys.platform == "linux":
        return Path(Path.home(), ".local", "share")
    elif sys.platform == "win32" or sys.platform == "cygwin":
        return Path(Path.home(), "AppData", "Roaming")
    elif sys.platform == "darwin":
        return Path(Path.home(), "Library", "Application Support")
    else:
        raise RuntimeError("Unsupported platform: " + sys.platform)
