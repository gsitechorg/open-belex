r"""
By Dylon Edwards
"""

from copy import deepcopy
from dataclasses import field
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Union

from open_belex.bleir.adapters import LocalToLoweredGroups
from open_belex.bleir.aggregators import CollectSpillRestoreCalls
from open_belex.bleir.allocators import (AllocateLoweredRegisters,
                                         AllocateRegisters)
from open_belex.bleir.analyzers import (CoalesceGroupedRegisters,
                                        CountNumInstructionsAndCommands,
                                        LiveParameterMarker,
                                        LiveSectionScanner,
                                        NumFragmentInstructionsAnalyzer,
                                        RegisterCoOcurrenceScanner,
                                        RegisterGrouper,
                                        RegisterParameterFinder,
                                        RegisterScanner, UserParameterScanner,
                                        UserRegisterScanner)
from open_belex.bleir.commands import CommandScheduler
from open_belex.bleir.generators import (BaryonHeaderFileWriter,
                                         BaryonHeaderGenerator,
                                         BaryonSourceFileWriter,
                                         BaryonSourceGenerator,
                                         ConstantsFileWriter,
                                         DeviceMainFileWriter,
                                         ExampleHeaderFileWriter,
                                         GvmlModuleGenerator,
                                         HostMainFileWriter,
                                         InternHeaderFileWriter,
                                         MesonFileWriter, TestHeaderFileWriter,
                                         TestSourceFileWriter,
                                         UtilsHeaderFileWriter,
                                         UtilsSourceFileWriter)
from open_belex.bleir.interpreters import BLEIRInterpreter
from open_belex.bleir.optimizers import (ConvergenceOptimizer,
                                         DoubleNegativeResolver,
                                         UnusedParameterRemover)
from open_belex.bleir.rewriters import (AllocateTemporaries, AutomaticLaner,
                                        CoalesceCompatibleTemporaries,
                                        CoalesceGroupedTemporaries,
                                        EnsureNoEmptyFragBodies,
                                        EnumerateInstructions,
                                        InitializeTemporaries,
                                        InjectKernelLibs, LongNymObfuscator,
                                        NormalizeSectionMasks,
                                        ParameterLowerizer,
                                        PartitionFragmentsIntoDigestibleChunks,
                                        ResetDebugValues,
                                        SpillRestoreScheduler)
from open_belex.bleir.semantic_validators import (
    EnsureWriteBeforeRead, FragmentCallerCallValidator,
    FragmentSignatureUniquenessValidator, MultiStatementSBGroupingEnforcer,
    MultiStatementValidator, NumFragmentInstructionsValidator,
    ParameterIDValidator, ReadWriteInhibitValidator, RegisterValidator,
    ReservedRegisterValidator, SnippetNameValidator)
from open_belex.bleir.syntactic_validators import (AssignmentPatternValidator,
                                                   BLEIRTypeValidator,
                                                   IdentifierValidator)
from open_belex.bleir.template_accessors import (AplTemplateAccessor,
                                                 BaryonTemplateAccessor,
                                                 MesonTemplateAccessor)
from open_belex.bleir.types import (BleirEnum, Fragment, FragmentCallerCall,
                                    Snippet, bleir_dataclass)
from open_belex.bleir.walkables import BLEIRWalker, Walkable
from open_belex.common.register_arenas import NUM_RN_REGS
from open_belex.common.stack_manager import StackManager
from open_belex.diri.half_bank import DIRI
from open_belex.utils.config_utils import CONFIG

Target = Union[Snippet, FragmentCallerCall, Fragment]

VIRTUAL_MACHINE = "VirtualMachine"


def noop(*args, **kwargs) -> None:
    pass


class Feature(BleirEnum):
    INJECT_KERNEL_LIBS: str = "inject-kernel-libs"
    RESET_DEBUG_VALUES: str = "reset-debug-values"
    INITIALIZE_TEMPORARIES: str = "initialize-temporaries"
    COALESCE_COMPATIBLE_TEMPORARIES: str = "coalesce-compatible-temporaries"
    SPILL_RESTORE_REGISTERS: str = "spill-restore-registers"
    NORMALIZE_SECTION_MASKS: str = "normalize-section-masks"
    ALLOCATE_TEMPORARIES: str = "allocate-temporaries"
    LOWER_PARAMETERS: str = "lower-parameters"
    ALLOCATE_REGISTERS: str = "allocate-registers"
    ALLOCATE_LOWERED_REGISTERS: str = "allocate-lowered-registers"
    REMOVE_UNUSED_PARAMETERS: str = "remove-unused-parameters"
    RESOLVE_DOUBLE_NEGATIVES: str = "resolve-double-negatives"
    AUTO_MERGE_COMMANDS: str = "auto-merge-commands"
    PARTITION_FRAGMENTS: str = "partition-fragments"
    ENUMERATE_INSTRUCTIONS: str = "enumerate-instructions"
    OBFUSCATE_LONG_NYMS: str = "obfuscate-long-nyms"
    INTERPRET_CODE: str = "interpret-code"
    GENERATE_CODE: str = "generate-code"


@bleir_dataclass
class BLEIRVirtualMachine:

    interpret: bool = False

    diri: Optional[DIRI] = None

    output_dir: Optional[Path] = None

    generate_code: bool = True

    generate_apl_sources: bool = True

    generate_test_app: bool = True

    generate_entry_point: bool = True

    uniquify_nyms: bool = True

    print_params: bool = False

    # Example: reservations = {"sm_regs": set(range(4, 15 + 1))}
    reservations: Optional[Dict[str, Set[int]]] = None

    features: Optional[Dict[Feature, bool]] = None
    extra_features: Optional[Dict[Feature, bool]] = None

    explicit_frags_only: bool = False

    explicit_frags_only: bool = False

    walker: BLEIRWalker = field(default_factory=BLEIRWalker)

    max_rn_regs: int = NUM_RN_REGS

    target: str = "baryon"

    _pipeline: Optional[Sequence[Walkable]] = None

    def __post_init__(self: "BLEIRVirtualMachine") -> None:
        if StackManager.has_elem(CONFIG):
            config = StackManager.peek(CONFIG)
        else:
            config = {}

        if self.reservations is not None:
            self.reservations = deepcopy(self.reservations)

        elif "reservations" in config:
            self.reservations = deepcopy(config["reservations"])

        else:
            self.reservations = dict()

        if "rn_regs" in config and "rn_regs" not in self.reervations:
            self.reservations["rn_regs"] = config["rn_regs"]

        for kind, vals in list(self.reservations.items()):
            if not isinstance(vals, set):
                self.reservations[kind] = set(vals)

        if "rn_regs" in self.reservations:
            reserved_rn_regs = set(self.reservations["rn_regs"])
            self.max_rn_regs = min(self.max_rn_regs,
                                   NUM_RN_REGS - len(reserved_rn_regs))

        if config is not None and "max_rn_regs" in config:
            self.max_rn_regs = min(self.max_rn_regs, config["max_rn_regs"])

        if "sm_regs" not in self.reservations:
            self.reservations["sm_regs"] = set()

        # Restore the default VM behavior
        if self.features is None:
            self.features = {feature: True for feature in Feature}
            del self.features[Feature.REMOVE_UNUSED_PARAMETERS]
            del self.features[Feature.OBFUSCATE_LONG_NYMS]
            del self.features[Feature.INTERPRET_CODE]
            del self.features[Feature.GENERATE_CODE]

        if Feature.OBFUSCATE_LONG_NYMS not in self.features:
            self.features[Feature.OBFUSCATE_LONG_NYMS] = self.uniquify_nyms

        if Feature.INTERPRET_CODE not in self.features:
            self.features[Feature.INTERPRET_CODE] = self.interpret

        if Feature.GENERATE_CODE not in self.features:
            self.features[Feature.GENERATE_CODE] = self.generate_code

        if self.extra_features is not None:
            self.features.update(self.extra_features)
            self.extra_features = None

        if self.explicit_frags_only:
            self.features[Feature.INJECT_KERNEL_LIBS] = False

    def has_feature(self: "BLEIRVirtualMachine", feature: Feature) -> bool:
        return feature in self.features and self.features[feature]

    @property
    def pipeline(self: "BLEIRVirtualMachine") -> Sequence[Walkable]:
        if self._pipeline is None:
            self.reset()
        return self._pipeline

    def reset(self: "BLEIRVirtualMachine") -> None:
        pipeline = []

        bleir_type_validator = BLEIRTypeValidator()
        pipeline.append(bleir_type_validator)

        assignment_pattern_validator = AssignmentPatternValidator()
        pipeline.append(assignment_pattern_validator)

        # inject_missing_noops = InjectMissingNOOPs()
        # pipeline.append(inject_missing_noops)

        if self.has_feature(Feature.INJECT_KERNEL_LIBS):
            inject_kernel_libs = InjectKernelLibs()
            pipeline.append(inject_kernel_libs)

        read_write_inhibit_validator = ReadWriteInhibitValidator()
        pipeline.append(read_write_inhibit_validator)

        if self.has_feature(Feature.RESET_DEBUG_VALUES):
            # House cleaning ...
            reset_debug_values = ResetDebugValues()
            pipeline.append(reset_debug_values)

        if self.has_feature(Feature.INITIALIZE_TEMPORARIES):
            initialize_temporaries = InitializeTemporaries()
            pipeline.append(initialize_temporaries)

        live_section_scanner = LiveSectionScanner()
        pipeline.append(live_section_scanner)

        ensure_write_before_read = EnsureWriteBeforeRead(live_section_scanner)
        pipeline.append(ensure_write_before_read)

        if self.has_feature(Feature.COALESCE_COMPATIBLE_TEMPORARIES):
            coalesce_compatible_temporaries = \
                CoalesceCompatibleTemporaries(live_section_scanner)
            pipeline.append(coalesce_compatible_temporaries)

        if self.has_feature(Feature.SPILL_RESTORE_REGISTERS):
            register_scanner = RegisterScanner(
                max_rn_regs=self.max_rn_regs)
            pipeline.append(register_scanner)

            register_co_ocurrence_scanner = RegisterCoOcurrenceScanner()
            pipeline.append(register_co_ocurrence_scanner)

            register_grouper = RegisterGrouper(
                register_scanner,
                register_co_ocurrence_scanner,
                max_rn_regs=self.max_rn_regs)
            pipeline.append(register_grouper)

            spill_restore_scheduler = SpillRestoreScheduler(
                register_scanner,
                register_grouper,
                max_rn_regs=self.max_rn_regs,
                reservations=self.reservations)
            pipeline.append(spill_restore_scheduler)

        if self.has_feature(Feature.NORMALIZE_SECTION_MASKS):
            normalize_section_masks = NormalizeSectionMasks(
                reservations=self.reservations["sm_regs"])
            pipeline.append(normalize_section_masks)

        if self.has_feature(Feature.ALLOCATE_TEMPORARIES):
            user_parameter_scanner = UserParameterScanner()
            pipeline.append(user_parameter_scanner)

            allocate_temporaries = AllocateTemporaries(
                user_parameter_scanner,
                reservations=self.reservations)
            pipeline.append(allocate_temporaries)

        if self.has_feature(Feature.LOWER_PARAMETERS):
            parameter_lowerizer = ParameterLowerizer()
            pipeline.append(parameter_lowerizer)

        if self.has_feature(Feature.SPILL_RESTORE_REGISTERS):
            local_to_lowered_groups = LocalToLoweredGroups(
                register_grouper,
                allocate_temporaries,
                parameter_lowerizer)

            coalesce_grouped_temporaries = CoalesceGroupedTemporaries(
                local_to_lowered_groups)
            pipeline.append(coalesce_grouped_temporaries)

            coalesce_grouped_registers = CoalesceGroupedRegisters(
                local_to_lowered_groups)
            shared_registers_by_frag = \
                coalesce_grouped_registers.shared_registers_by_frag
            pipeline.append(coalesce_grouped_registers)
        else:
            shared_registers_by_frag = {}

        if self.has_feature(Feature.ALLOCATE_REGISTERS):
            allocate_registers = AllocateRegisters(
                shared_registers_by_frag,
                reservations=self.reservations)
            pipeline.append(allocate_registers)

        if self.has_feature(Feature.LOWER_PARAMETERS):
            # Second pass of parameter_lowerizer to remove lowered registers from
            # fragment callers
            pipeline.append(parameter_lowerizer)

        if self.has_feature(Feature.SPILL_RESTORE_REGISTERS):
            local_to_lowered_groups = LocalToLoweredGroups(
                register_grouper,
                allocate_temporaries,
                parameter_lowerizer,
                coalesce_grouped_registers)
        else:
            local_to_lowered_groups = noop

        if self.has_feature(Feature.ALLOCATE_LOWERED_REGISTERS):
            user_register_scanner = UserRegisterScanner(
                local_to_lowered_groups,
                shared_registers_by_frag
            )
            pipeline.append(user_register_scanner)

            allocate_lowered_registers = AllocateLoweredRegisters(
                user_register_scanner,
                shared_registers_by_frag,
                reservations=self.reservations)
            pipeline.append(allocate_lowered_registers)

        if self.has_feature(Feature.ALLOCATE_REGISTERS):
            pipeline.append(allocate_registers)

        # rewrite_singleton_multi_statement = \
        #     RewriteSingletonMultiStatement()
        # pipeline.append(rewrite_singleton_multi_statement)

        snippet_name_validator = SnippetNameValidator()
        pipeline.append(snippet_name_validator)

        parameter_id_validator = ParameterIDValidator()
        pipeline.append(parameter_id_validator)

        if self.has_feature(Feature.ALLOCATE_REGISTERS):
            register_validator = RegisterValidator()
            pipeline.append(register_validator)

        fragment_caller_call_validator = FragmentCallerCallValidator()
        pipeline.append(fragment_caller_call_validator)

        fragment_signature_uniqueness_validator = \
            FragmentSignatureUniquenessValidator()
        pipeline.append(fragment_signature_uniqueness_validator)

        reserved_register_validator = ReservedRegisterValidator()
        pipeline.append(reserved_register_validator)

        multi_statement_grouping_enforcer = MultiStatementSBGroupingEnforcer()
        pipeline.append(multi_statement_grouping_enforcer)

        # allocated_register_uniqueness_validator = \
        #     AllocatedRegisterUniquenessValidator()
        # pipeline.append(allocated_register_uniqueness_validator)

        ## ======================== ##
        ## BEGIN: Optimization Loop ##
        ## =================================================================== ##

        optimizers = []

        if self.has_feature(Feature.REMOVE_UNUSED_PARAMETERS):
            # It's not an optimizer, but it is important to update its internal
            # stats on each pass to avoid removing incorrect parameters with
            # the unused_parameter_remover
            live_parameters_marker = LiveParameterMarker()
            optimizers.append(live_parameters_marker)

            unused_parameter_remover = \
                UnusedParameterRemover(live_parameters_marker)
            optimizers.append(unused_parameter_remover)

        if self.has_feature(Feature.RESOLVE_DOUBLE_NEGATIVES):
            double_negative_resolver = DoubleNegativeResolver()
            optimizers.append(double_negative_resolver)

        if len(optimizers) > 0:
            convergence_optimizer = \
                ConvergenceOptimizer(optimizers=optimizers, walker=self.walker)
            pipeline.append(convergence_optimizer)

        ## =================================================================== ##
        ## END: Optimization Loop ##
        ## ====================== ##

        if self.has_feature(Feature.AUTO_MERGE_COMMANDS):
            automatic_laner = AutomaticLaner()
            pipeline.append(automatic_laner)

        num_fragment_instructions_analyzer = \
            NumFragmentInstructionsAnalyzer()
        pipeline.append(num_fragment_instructions_analyzer)

        if self.has_feature(Feature.PARTITION_FRAGMENTS):
            partition_fragments_into_digestible_chunks = \
                PartitionFragmentsIntoDigestibleChunks(
                    num_fragment_instructions_analyzer)
            pipeline.append(partition_fragments_into_digestible_chunks)

        if self.has_feature(Feature.ENUMERATE_INSTRUCTIONS):
            enumerate_instructions = EnumerateInstructions()
            pipeline.append(enumerate_instructions)

        if self.has_feature(Feature.OBFUSCATE_LONG_NYMS):
            long_nym_obfuscator = LongNymObfuscator()
            pipeline.append(long_nym_obfuscator)

        if pipeline[-1] is not num_fragment_instructions_analyzer:
            # Analyze the number of instructions again, after partitioning
            pipeline.append(num_fragment_instructions_analyzer)

        ensure_no_empty_frag_bodies = EnsureNoEmptyFragBodies(
            num_fragment_instructions_analyzer=num_fragment_instructions_analyzer)
        pipeline.append(ensure_no_empty_frag_bodies)

        if self.has_feature(Feature.OBFUSCATE_LONG_NYMS):
            identifier_validator = IdentifierValidator()
            pipeline.append(identifier_validator)

        multi_statement_validator = MultiStatementValidator()
        pipeline.append(multi_statement_validator)

        num_fragment_instructions_validator = \
            NumFragmentInstructionsValidator(
                num_fragment_instructions_analyzer)
        pipeline.append(num_fragment_instructions_validator)

        register_parameter_finder = RegisterParameterFinder()
        pipeline.append(register_parameter_finder)

        if self.has_feature(Feature.SPILL_RESTORE_REGISTERS):
            collect_spill_restore_calls = CollectSpillRestoreCalls(
                spill_restore_scheduler,
                register_grouper,
                allocate_temporaries,
                parameter_lowerizer,
                coalesce_grouped_temporaries,
                coalesce_grouped_registers)
            pipeline.append(collect_spill_restore_calls)

        if self.has_feature(Feature.INTERPRET_CODE):
            interpreter = \
                BLEIRInterpreter(
                    coalesce_grouped_registers,
                    diri=self.diri)
            pipeline.append(interpreter)

        if self.has_feature(Feature.GENERATE_CODE):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            apl_template_accessor = AplTemplateAccessor()

            if self.target == "baryon":
                if self.generate_apl_sources:
                    baryon_template_accessor = BaryonTemplateAccessor()

                    command_scheduler = CommandScheduler()
                    pipeline.append(command_scheduler)

                    baryon_header_generator = \
                        BaryonHeaderGenerator(
                            baryon_template_accessor,
                            apl_template_accessor,
                            register_parameter_finder,
                            explicit_frags_only=self.explicit_frags_only,
                            target=self.target)

                    baryon_header_file_writer = \
                        BaryonHeaderFileWriter(self.output_dir,
                                               baryon_header_generator)
                    pipeline.append(baryon_header_file_writer)

                    generate_belex_callers = \
                        self.has_feature(Feature.ALLOCATE_REGISTERS)

                    baryon_source_generator = \
                        BaryonSourceGenerator(
                            baryon_template_accessor,
                            command_scheduler,
                            apl_template_accessor,
                            register_parameter_finder,
                            generate_belex_callers=generate_belex_callers,
                            explicit_frags_only=self.explicit_frags_only)

                    baryon_source_file_writer = \
                        BaryonSourceFileWriter(self.output_dir,
                                               baryon_source_generator)
                    pipeline.append(baryon_source_file_writer)

            if not self.explicit_frags_only:
                if self.generate_apl_sources:
                    constants_file_writer = \
                        ConstantsFileWriter(self.output_dir,
                                            apl_template_accessor)
                    pipeline.append(constants_file_writer)

                    utils_source_file_writer = \
                        UtilsSourceFileWriter(self.output_dir,
                                              apl_template_accessor)
                    pipeline.append(utils_source_file_writer)

                    utils_header_file_writer = \
                        UtilsHeaderFileWriter(self.output_dir,
                                              apl_template_accessor)
                    pipeline.append(utils_header_file_writer)

                if self.generate_test_app:
                    intern_header_file_writer = \
                        InternHeaderFileWriter(self.output_dir,
                                               apl_template_accessor)
                    pipeline.append(intern_header_file_writer)

                    example_header_file_writer = \
                        ExampleHeaderFileWriter(self.output_dir,
                                                apl_template_accessor)
                    pipeline.append(example_header_file_writer)

                    gvml_module_generator = \
                        GvmlModuleGenerator(apl_template_accessor)

                    device_main_file_writer = \
                        DeviceMainFileWriter(self.output_dir,
                                             gvml_module_generator)
                    pipeline.append(device_main_file_writer)

                    test_header_file_writer = \
                        TestHeaderFileWriter(self.output_dir,
                                             apl_template_accessor)
                    pipeline.append(test_header_file_writer)

                    count_num_instructions_and_commands = \
                        CountNumInstructionsAndCommands(
                            num_fragment_instructions_analyzer)
                    pipeline.append(count_num_instructions_and_commands)

                    test_source_file_writer = \
                        TestSourceFileWriter(self.output_dir,
                                             apl_template_accessor,
                                             count_num_instructions_and_commands)
                    pipeline.append(test_source_file_writer)

                    host_main_file_writer = \
                        HostMainFileWriter(self.output_dir,
                                           apl_template_accessor,
                                           count_num_instructions_and_commands)
                    pipeline.append(host_main_file_writer)

                    if self.generate_entry_point:
                        meson_template_accessor = MesonTemplateAccessor()
                        meson_file_writer = \
                            MesonFileWriter(
                                self.output_dir,
                                meson_template_accessor,
                                target=self.target)
                        pipeline.append(meson_file_writer)

        self._pipeline = pipeline

    def compile(self: "BLEIRVirtualMachine", target: Target) -> Target:
        if isinstance(target, Snippet):
            self.reset()

        for stage in self.pipeline:
            target = self.walker.walk(stage, target)

        return target

    @property
    def interpreter(self: "BLEIRVirtualMachine") -> Optional[BLEIRInterpreter]:
        for stage in reversed(self.pipeline):
            if isinstance(stage, BLEIRInterpreter):
                return stage
        return None

    def assert_no_interpreter_failures(self: "BLEIRVirtualMachine") -> None:
        """Helper method added to ensure code is always generated even if a
        unit test fails in the interpreter. This method should be called after
        compiling a Snippet to check for unit test failures from the
        interpreter."""
        if self.interpreter is not None and len(self.interpreter.failures) > 0:
            message, *_ = self.interpreter.failures[0]
            raise AssertionError(message)
