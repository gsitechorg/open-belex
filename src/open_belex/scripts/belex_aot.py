r"""
By Dylon Edwards
"""

import logging
import logging.handlers
import sys
from itertools import chain

import click

from open_belex.apl_optimizations import APL_OPTIMIZATIONS
from open_belex.bleir.types import FragmentCaller, Snippet, SnippetMetadata
from open_belex.bleir.virtual_machines import BLEIRVirtualMachine, Feature
from open_belex.utils.log_utils import LogLevel, init_logger
from open_belex.utils.script_utils import (collect_config, collect_header_file,
                                           collect_log_level,
                                           collect_optimizations,
                                           collect_source_file,
                                           enable_optimizations_by_level,
                                           for_each_belex_caller,
                                           target_source_and_header_exts)

SCRIPT_NAME = "belex-aot"

LOGGER = logging.getLogger()


@click.command()
@click.option("-O", "--optimization-level", "optimization_level",
              help="Specifies the optimization level.",
              type=click.IntRange(0, 4), default=2, show_default=True)
@click.option("-f", "--enable-optimization", "optimizations",
              help="Specifies an optimization over the generated code. Many "
                   "optimizations apply only to high-level BELEX.",
              type=click.Choice(list(chain(APL_OPTIMIZATIONS.keys(),
                                           Feature.values(),
                                           [f"no-{feature}"
                                            for feature in Feature]))),
              multiple=True,
              callback=collect_optimizations)
@click.option("-b", "--belex-dir", "belex_dir",
              help="Path to the folder containing the BELEX files.",
              type=click.Path(exists=True, file_okay=False),
              required=False)
@click.option("-s", "--source-file", "source_file",
              help="Path to the source file to generate.",
              type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("-h", "--header-file", "header_file",
              help="Path to the header file to generate.",
              type=click.Path(exists=False, file_okay=True, dir_okay=False),
              required=False)
@click.option("--target", "target",
              help="Type of sources to generate.",
              type=click.Choice(["baryon"]),
              default="baryon",
              required=False)
@click.option("--high-level/--no-high-level", "include_high_level",
              help="Whether to include high-level BELEX frags.",
              default=True)
@click.option("--low-level/--no-low-level", "include_low_level",
              help="Whether to include low-level BELEX frags.",
              default=True)
@click.option("--uniquify-nyms/--no-uniquify-nyms", "uniquify_nyms",
              help=("Whether to ensure uniqueness of long C-identifiers by "
                    "obfuscating them to have less than 31 chars."),
              default=False)
@click.option("--config", "config",
              help="Specifies path to the BELEX config YAML file.",
              callback=collect_config,
              required=False)
@click.option("--log-level", "log_level",
              help="Specifies the verbosity of output from the compiler.",
              type=click.Choice(LogLevel.names()),
              default=LogLevel.DEFAULT.name,
              callback=collect_log_level,
              required=False)
@click.option("--explicit-frags-only/--generate-all-sources",
              "explicit_frags_only",
              help=("Generates only the fragments that appear in the source; "
                    "none of the helper functions are generated. Only use "
                    "this feature if you provide the missing support "
                    "functions."),
              default=False)
@click.argument("belex_files", nargs=-1)
@enable_optimizations_by_level
@collect_source_file
@collect_header_file
def main(**kwargs) -> None:
    """Generates APL and headers files with the given fragments."""

    global LOGGER, SCRIPT_NAME

    log_level = kwargs["log_level"]
    init_logger(LOGGER, SCRIPT_NAME, log_level=log_level)

    for arg, val in kwargs.items():
        LOGGER.debug("%s = %s", arg, val)

    source_file = kwargs["source_file"]
    header_file = kwargs["header_file"]

    output_dir = source_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    source_ext, header_ext = \
        target_source_and_header_exts(kwargs["target"])
    snippet_name = source_file.name[:-len(source_ext)]

    library_callers = []

    def callback(_, belex_caller):
        nonlocal library_callers
        if not isinstance(belex_caller, FragmentCaller) \
           and hasattr(belex_caller, "__caller__") \
           and isinstance(belex_caller.__caller__, FragmentCaller):
            belex_caller = belex_caller.__caller__
        library_callers.append(belex_caller)

    for_each_belex_caller(kwargs["belex_dir"],
                          kwargs["belex_files"],
                          callback,
                          kwargs["optimizations"]["high-level"],
                          kwargs["include_high_level"],
                          kwargs["include_low_level"])

    snippet = Snippet(
        name=snippet_name,
        examples=[],
        calls=[],
        library_callers=library_callers,
        metadata={
            SnippetMetadata.HEADER_FILE: header_file,
            SnippetMetadata.SOURCE_FILE: source_file,
            SnippetMetadata.TARGET: kwargs["target"],
        })

    config = kwargs["config"]
    virtual_machine = BLEIRVirtualMachine(
        output_dir=output_dir,
        interpret=False,
        generate_code=True,
        generate_apl_sources=True,
        generate_test_app=False,
        uniquify_nyms=kwargs["uniquify_nyms"],
        reservations=config["reservations"],
        features=kwargs["optimizations"]["low-level"],
        explicit_frags_only=kwargs["explicit_frags_only"],
        target=kwargs["target"])

    virtual_machine.compile(snippet)
    virtual_machine.assert_no_interpreter_failures()
    LOGGER.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Failed to generate BELEX examples")
        sys.exit(1)
