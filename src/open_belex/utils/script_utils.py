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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from getpass import getuser
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmembers, isfunction
from pathlib import Path
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Set, Tuple, Type, Union)

import click

from open_belex.apl_optimizations import APL_OPTIMIZATIONS
from open_belex.bleir.template_accessors import MesonTemplateAccessor
from open_belex.bleir.types import FragmentCaller
from open_belex.bleir.virtual_machines import Feature
from open_belex.utils.config_utils import load_config
from open_belex.utils.log_utils import LogLevel

LOGGER = logging.getLogger()


def collect_log_level(ctx: click.Context,
                      option: click.Option,
                      log_level: str) -> int:
    log_level = LogLevel[log_level]
    return log_level.value


class Reservations(Enum):
    GVML = "GVML"
    NONE = "NONE"

    @classmethod
    def names(cls: Type["Reservations"]) -> Sequence[str]:
        names = []
        for reservations in cls:
            names.append(reservations.name)
        return names


RESERVATIONS: Dict[Reservations, Dict[str, Set[int]]] = {
    Reservations.GVML: {
        "row_numbers": set(range(15, 23 + 1)),
        "sm_regs": set(range(4, 15 + 1)),
    },
    Reservations.NONE: {
    },
}


def collect_config(ctx: click.Context,
                   option: click.Option,
                   config_path: str) -> Dict[str, Set[int]]:

    config = load_config(config_path)
    if "reservations" in config:
        reservations = config["reservations"]
        for reservation_kind, reservation_list in list(reservations.items()):
            reservations[reservation_kind] = set(reservation_list)
    else:
        config["reservations"] = {}
    return config


def collect_reservations(ctx: click.Context,
                         option: click.Option,
                         reservations: str) -> Dict[str, Set[int]]:

    global RESERVATIONS

    if reservations in Reservations.names():
        reservations = Reservations[reservations]
        return RESERVATIONS[reservations]

    config_path = reservations
    config = load_config(config_path)
    reservations = config["reservations"]
    for reservation_kind, reservation_list in list(reservations.items()):
        reservations[reservation_kind] = set(reservation_list)
    return reservations


def generate_belex_examples_meson_build(output_dir: Path,
                                        example_dirs: Sequence[Path],
                                        manifest: Dict[str, Path]) -> None:
    global LOGGER
    output_dir.mkdir(parents=True, exist_ok=True)
    meson_build_script = output_dir / "meson.build"
    LOGGER.info(f"Generating: {meson_build_script} ...")
    meson_template_accessor = MesonTemplateAccessor()
    curr_time = datetime.now()
    script_body = \
        meson_template_accessor.emit_belex_examples_meson_build(
            username=getuser(),
            timestamp=curr_time.isoformat(),
            example_dirs=example_dirs,
            manifest=manifest)
    script_body = f"{script_body}\n"
    if meson_build_script.exists():
        with open(meson_build_script, "rt") as f:
            if script_body == f.read():
                return
    with open(meson_build_script, "wt") as f:
        f.write(script_body)


@dataclass
class OptimizationsBuilder:
    optimizations_and_features: Optional[Dict[str, Any]] = None
    high_level_optimizations: List[Callable] = field(default_factory=list)
    low_level_features: Dict[Feature, bool] = field(default_factory=dict)

    def __post_init__(self: "OptimizationsBuilder") -> None:
        if self.optimizations_and_features is not None:
            high_level_optimizations = self.optimizations_and_features["high-level"]
            low_level_features = self.optimizations_and_features["low-level"]
            self.high_level_optimizations.extend(high_level_optimizations)
            self.low_level_features.update(low_level_features)

    def add(self: "OptimizationsBuilder", opt_nym: str) -> None:
        if opt_nym in APL_OPTIMIZATIONS:
            optimization = APL_OPTIMIZATIONS[opt_nym]
            if optimization not in self.high_level_optimizations:
                self.high_level_optimizations.append(optimization)

        else:
            feature = opt_nym
            if feature.startswith("no-"):
                feature = feature[len("no-"):]
                has_feature = False
            else:
                has_feature = True

            feature = Feature.find_by_value(feature)
            if feature not in self.low_level_features:
                self.low_level_features[feature] = has_feature

    def as_dict(self: "OptimizationsBuilder") -> Dict[str, Any]:
        optimizations_by_level = {
            "high-level": self.high_level_optimizations,
            "low-level": self.low_level_features,
        }

        return optimizations_by_level


def collect_optimizations(
        ctx: click.Context,
        option: click.Option,
        optimizations: Sequence[str]) -> Dict[str,
                                              Union[Sequence[Callable],
                                                    Dict[str, bool]]]:

    optimizations_builder = OptimizationsBuilder()

    for optimization in optimizations:
        optimizations_builder.add(optimization)

    optimizations_by_level = optimizations_builder.as_dict()
    return optimizations_by_level


def optimizations_by_level(
            optimization_level: int,
            optimizations_and_features: Optional[Dict[Any, Any]] = None) \
        -> Dict[Any, Any]:

    optimizations_builder = OptimizationsBuilder(
        optimizations_and_features=optimizations_and_features)

    if optimization_level >= 1:
        # High-level optimizations
        optimizations_builder.add("replace-zero-xor")

        # Low-level features
        optimizations_builder.add("inject-kernel-libs")
        optimizations_builder.add("reset-debug-values")
        optimizations_builder.add("allocate-registers")
        optimizations_builder.add("partition-fragments")
        optimizations_builder.add("initialize-temporaries")
        optimizations_builder.add("normalize-section-masks")
        optimizations_builder.add("allocate-temporaries")
        optimizations_builder.add("lower-parameters")
        optimizations_builder.add("allocate-lowered-registers")

    if optimization_level >= 2:
        # High-level optimizations
        optimizations_builder.add("eliminate-read-after-write")
        optimizations_builder.add("delete-dead-writes")
        optimizations_builder.add("coalesce-consecutive-writes")
        optimizations_builder.add("coalesce-consecutive-and-reads")
        optimizations_builder.add("coalesce-sb-from-src")
        optimizations_builder.add("coalesce-sb-from-rl")
        optimizations_builder.add("merge-rl-src-sb")
        optimizations_builder.add("merge-rl-src-sb2")
        optimizations_builder.add("eliminate-write-read-dependence")

        # Low-level features
        optimizations_builder.add("coalesce-compatible-temporaries")
        optimizations_builder.add("spill-restore-registers")
        optimizations_builder.add("resolve-double-negatives")
        optimizations_builder.add("auto-merge-commands")
        optimizations_builder.add("enumerate-instructions")

    if optimization_level >= 3:
        # Low-level features
        optimizations_builder.add("remove-unused-parameters")

    return optimizations_builder.as_dict()


def enable_optimizations_by_level(fn: Callable) -> Callable:

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "optimization_level" in kwargs and "optimizations" in kwargs:
            optimization_level = kwargs["optimization_level"]
            optimizations_and_features = kwargs["optimizations"]
            kwargs["optimizations"] = optimizations_by_level(
                optimization_level,
                optimizations_and_features=optimizations_and_features)

        retval = fn(*args, **kwargs)
        return retval

    return wrapper


def collect_belex_files(belex_dir: Path) -> Iterator[Path]:
    for candidate_path in belex_dir.iterdir():
        if candidate_path.is_file() and candidate_path.name.endswith(".py"):
            belex_file = candidate_path
            yield belex_file


def collect_belex_callers(belex_file: Path,
                          optimizations: Sequence[Callable],
                          include_high_level: bool = True,
                          include_low_level: bool = True,
                          include_tests: bool = False) \
                          -> Iterator[Union[FragmentCaller, Callable]]:

    global LOGGER

    belex_spec = spec_from_file_location("belex_spec", belex_file)
    belex_module = module_from_spec(belex_spec)
    belex_spec.loader.exec_module(belex_module)

    for _, fn in getmembers(belex_module, isfunction):
        fragment_caller = None

        outer_fn = None
        if hasattr(fn, "is_hypothesis_test"):
            outer_fn = fn
            fn = fn.hypothesis.inner_test

        if include_high_level \
           and hasattr(fn, "__high_level_block__") \
           and not hasattr(fn, "__skip_tests__"):
            # fragment_caller_call = fn(optimizations=optimizations)
            # fragment_caller = fragment_caller_call.caller
            # yield fragment_caller
            yield fn

        elif include_high_level and include_tests \
             and hasattr(fn, "__high_level_test__"):
            yield fn

        elif include_low_level and hasattr(fn, "__low_level_block__"):
            fragment_caller = None
            if hasattr(fn, "__caller__"):
                fragment_caller = fn.__caller__

            if not isinstance(fragment_caller, FragmentCaller) \
               and callable(fragment_caller):
                build_fragment_caller = fragment_caller
                fragment_caller = build_fragment_caller()

            if isinstance(fragment_caller, FragmentCaller):
                yield fragment_caller
            else:
                raise RuntimeError(
                    f"Unknown form of low-level BELEX caller in {belex_file}: "
                    f"{fn.__name__}")

        elif include_low_level and include_tests \
             and hasattr(fn, "__low_level_test__"):
            if outer_fn is not None:
                yield outer_fn
            else:
                yield fn


def for_each_belex_caller(belex_dir: Optional[Path],
                          belex_files: Sequence[str],
                          fn: Callable[[FragmentCaller], None],
                          optimizations: Sequence[Callable],
                          include_high_level: bool = True,
                          include_low_level: bool = True,
                          include_tests: bool = False) -> None:

    global LOGGER

    if belex_dir is not None:
        belex_dir = Path(belex_dir)
        LOGGER.info(f"Scanning for BELEX fragments in {belex_dir} ...")
        for belex_file in collect_belex_files(belex_dir):
            LOGGER.info(f"Scanning for BELEX fragments in {belex_file} ...")
            num_callers = 0
            for belex_caller_or_test_fn in \
                collect_belex_callers(belex_file, optimizations,
                                      include_high_level,
                                      include_low_level,
                                      include_tests):
                try:
                    if isinstance(belex_caller_or_test_fn, FragmentCaller):
                        belex_caller = belex_caller_or_test_fn
                        fragment = belex_caller.fragment
                        LOGGER.info(f"Found {fragment.identifier} in "
                                    f"{belex_file}.")
                        fn(belex_file, belex_caller)
                    else:
                        belex_test_fn = belex_caller_or_test_fn
                        LOGGER.info(f"Found {belex_test_fn.__name__} in "
                                    f"{belex_file}.")
                        fn(belex_test_fn, belex_test_fn)
                    num_callers += 1
                except Exception as exception:
                    raise RuntimeError(
                        f"Failed to process [{belex_caller_or_test_fn}] from: "
                        f"{belex_file}") from exception
            LOGGER.info(f"Found {num_callers} BELEX fragments in {belex_file}")

    for belex_file in belex_files:
        belex_parts = belex_file.split("::")
        belex_file = Path(belex_parts[0])
        if not belex_file.exists():
            raise AssertionError(f"BELEX file does not exist: {belex_file}")

        specific_fragment = None
        if len(belex_parts) == 2:
            specific_fragment = belex_parts[1]
            LOGGER.info(f"Scanning for BELEX fragments, {specific_fragment}, "
                        f"in {belex_file} ...")
        elif len(belex_parts) > 2:
            raise AssertionError(f"Only one caller may be specified per file: "
                                 f"{belex_parts}")
        else:
            LOGGER.info(f"Scanning for BELEX fragments in {belex_file} ...")

        found_fragment = False
        for belex_caller_or_test_fn in \
            collect_belex_callers(belex_file, optimizations,
                                  include_high_level,
                                  include_low_level,
                                  include_tests):
            if isinstance(belex_caller_or_test_fn, FragmentCaller):
                belex_caller = belex_caller_or_test_fn
                fragment = belex_caller.fragment
                if specific_fragment is None \
                   or specific_fragment == fragment.identifier:
                    try:
                        LOGGER.info(f"Found {fragment.identifier} in "
                                    f"{belex_file}.")
                        found_fragment = True
                        fn(belex_file, belex_caller)
                    except Exception as exception:
                        raise RuntimeError(
                            f"Failed to process "
                            f"[{belex_caller.fragment.identifier}] from: "
                            f"{belex_file}") from exception
            else:
                belex_test_fn = belex_caller_or_test_fn
                if specific_fragment is None \
                   or specific_fragment == belex_test_fn.__name__:
                    try:
                        LOGGER.info(f"Found {belex_test_fn.__name__} in "
                                    f"{belex_file}.")
                        found_fragment = True
                        fn(belex_test_fn, belex_test_fn)
                    except Exception as exception:
                        raise RuntimeError(
                            f"Failed to process [{belex_test_fn.__name__}] "
                            f"from: {belex_file}") from exception

        if specific_fragment is not None and not found_fragment:
            raise ValueError(f"Failed to find fragment {specific_fragment} in "
                             f"{belex_file}")


# See: https://stackoverflow.com/a/50491613/206543
class DefaultHelp(click.Command):

    def __init__(self, *args, **kwargs):
        context_settings = kwargs.setdefault('context_settings', {})
        if 'help_option_names' not in context_settings:
            context_settings['help_option_names'] = ['-h', '--help']
        self.help_flag = context_settings['help_option_names'][0]
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        if not args:
            args = [self.help_flag]
        return super().parse_args(ctx, args)


def target_source_and_header_exts(target: str) -> Tuple[str, str]:
    if target == "baryon":
        source_ext = ".c"
        header_ext = ".h"
    else:
        raise RuntimeError(
            f"Unsupported target: {target}")
    return source_ext, header_ext


def collect_source_file(fn: Callable) -> Callable:

    def get_or_build_source_file(source_file: Optional[str],
                                 header_file: Optional[str],
                                 source_ext: str,
                                 header_ext: str) -> Path:
        if source_file is not None:
            source_file = Path(source_file)
        elif header_file is not None and header_file.endswith(header_ext):
            source_file = header_file[:-len(header_ext)]
            source_file = Path(f"{source_file}{source_ext}")
        else:
            raise ValueError(
                "A valid source_file or header_file must be specified.")

        if not source_file.name.endswith(source_ext):
            raise ValueError(
                f"source_file name must end with {source_ext}: "
                f"{source_file.name}")

        return source_file

    @wraps(fn)
    def wrapper(**kwargs) -> None:
        source_ext, header_ext = \
            target_source_and_header_exts(kwargs["target"])
        source_file = get_or_build_source_file(kwargs["source_file"],
                                               kwargs["header_file"],
                                               source_ext,
                                               header_ext)
        kwargs["source_file"] = source_file
        fn(**kwargs)

    return wrapper


def collect_header_file(fn: Callable) -> Callable:

    def get_or_build_header_file(source_file: Path,
                                 header_file: Optional[str],
                                 source_ext: str,
                                 header_ext: str) -> Path:
        if header_file is None:
            output_dir = source_file.parent
            snippet_name = source_file.name[:-len(source_ext)]
            header_file = output_dir / f"{snippet_name}{header_ext}"
        else:
            header_file = Path(header_file)
            if not header_file.name.endswith(header_ext):
                raise ValueError(
                    f"header_file name must end with {header_ext}: {header_file.name}")

        return header_file

    @wraps(fn)
    def wrapper(**kwargs) -> None:
        source_ext, header_ext = \
            target_source_and_header_exts(kwargs["target"])
        header_file = get_or_build_header_file(kwargs["source_file"],
                                               kwargs["header_file"],
                                               source_ext,
                                               header_ext)
        kwargs["header_file"] = header_file
        fn(**kwargs)

    return wrapper
