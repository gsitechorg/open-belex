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

import logging
import logging.handlers
import sys

import click

from open_belex.apl_optimizations import APL_OPTIMIZATIONS
from open_belex.bleir.pretty_printer import pretty_print
from open_belex.bleir.types import FragmentCaller
from open_belex.utils.log_utils import LogLevel, init_logger
from open_belex.utils.script_utils import (DefaultHelp, collect_log_level,
                                           collect_optimizations,
                                           for_each_belex_caller)

SCRIPT_NAME = "belex-pp"

LOGGER = logging.getLogger()


@click.command(cls=DefaultHelp)
@click.option("-f", "--enable-optimization", "optimizations",
              help="Specifies an optimization over the generated code. Many "
                   "optimizations apply only to high-level BELEX.",
              type=click.Choice(APL_OPTIMIZATIONS.keys()),
              multiple=True,
              callback=collect_optimizations)
@click.option("-b", "--belex-dir", "belex_dir",
              help="Path to the folder containing the BELEX files.",
              type=click.Path(exists=True, file_okay=False),
              required=False)
@click.option("--color/--no-color", "colorize",
              help="Whether to colorize the output.",
              default=True)
@click.option("--high-level/--no-high-level", "include_high_level",
              help="Whether to include high-level BELEX frags. [Default: true]",
              default=True)
@click.option("--low-level/--no-low-level", "include_low_level",
              help="Whether to include low-level BELEX frags. [Default: true]",
              default=True)
@click.option("--log-level", "log_level",
              help="Specifies the verbosity of output from the compiler.",
              type=click.Choice(LogLevel.names()),
              default=LogLevel.DEFAULT.name,
              callback=collect_log_level,
              required=False)
@click.argument("belex_files", nargs=-1)
def main(**kwargs):

    """Pretty-prints BELEX fragments.

    Example Usage:

        belex-pp path/to/src/library.py::belex_frag

        belex-pp path/to/src/library.py

        belex-pp path/to/src/"""

    global LOGGER, SCRIPT_NAME
    init_logger(LOGGER, SCRIPT_NAME, log_level=kwargs["log_level"])

    for arg, val in kwargs.items():
        LOGGER.debug("%s = %s", arg, val)

    def callback(_, belex_caller):
        nonlocal kwargs
        if not isinstance(belex_caller, FragmentCaller) \
           and hasattr(belex_caller, "__caller__") \
           and isinstance(belex_caller.__caller__, FragmentCaller):
            belex_caller = belex_caller.__caller__
        pretty_print(belex_caller, colorize=kwargs["colorize"])

    for_each_belex_caller(kwargs["belex_dir"],
                          kwargs["belex_files"],
                          callback,
                          kwargs["optimizations"],
                          kwargs["include_high_level"],
                          kwargs["include_low_level"])

    LOGGER.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Failed to generate BELEX examples")
        sys.exit(1)

