import re
from pathlib import Path
from re import Pattern
from typing import Sequence, Union

from setuptools import setup

def recursively_collect_data_files(
        directory: Union[str, Path],
        pattern: Union[str, Pattern]) -> Sequence[Path]:

    if isinstance(directory, str):
        directory = Path(directory)

    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    global_data_files = []
    local_data_files = []
    for candidate in directory.iterdir():
        if candidate.is_dir():
            global_data_files += recursively_collect_data_files(candidate, pattern)
        elif pattern.fullmatch(candidate.name):
            local_data_files.append(str(candidate))

    if len(local_data_files) > 0:
        global_data_files.append((f"share/belex/{directory}", local_data_files))

    return global_data_files

setup(
    data_files=recursively_collect_data_files("templates", r".*\.jinja"),
)
