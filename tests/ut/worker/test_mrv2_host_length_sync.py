# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute MRV2's length consumer without importing NPU/vLLM dependencies.

Run directly with Python's unittest runner on a host without PyTorch. Device
snapshots deliberately reject reads before their completion event is waited on.
"""

import __future__

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def load_length_consumer():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in runner.body if getattr(node, "name", None) == "_update_seq_lens_cpu")
    namespace = {"is_deepseek_v41": lambda config: config.model_type == "deepseek_v41"}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec", __future__.annotations.compiler_flag),
        namespace,
    )
    return namespace["_update_seq_lens_cpu"]


def load_snapshot_runner():
    """Run the real capability property and runner methods with CPU state."""
    repo = Path(__file__).resolve().parents[3]
    runner_path = repo / "vllm_ascend/worker/v2/model_runner.py"
    draft_path = repo / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
    definitions = []
    for path, name, methods, bases in (
        (draft_path, "AscendDSparkSpeculator", {"requires_target_cpu_seq_lens"}, []),
        (
            runner_path,
            "NPUModelRunner",
            {
                "_can_skip_spec_cpu_length_sync",
                "_copy_num_computed_tokens_to_cpu",
                "_update_seq_lens_cpu",
                "postprocess_sampled",
                "postprocess_num_computed_tokens",
            },
            [ast.Name(id="CPUParent", ctx=ast.Load())],
        ),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        definition.body = [node for node in definition.body if getattr(node, "name", None) in methods]
        definition.bases = bases
        definitions.append(definition)

    class CPUParent:
        def postprocess_sampled(self, idx_mapping, tokens, num_sampled, num_rejected, query_start_loc):
            self.parent_calls.append("reject")
            for index, rejected in zip(idx_mapping, num_rejected):
                self.req_states.num_computed_tokens.gpu[index] -= rejected

        def postprocess_num_computed_tokens(self, batch):
            self.parent_calls.append("advance")
            for index, count in zip(batch.idx_mapping, batch.num_scheduled_tokens):
                self.req_states.num_computed_tokens.gpu[index] += count

    backend = SimpleNamespace(
        cuda=SimpleNamespace(current_stream=Mock(side_effect=AssertionError("unexpected target D2H"))),
        npu=SimpleNamespace(stream=Mock(side_effect=AssertionError("unexpected target copy stream"))),
    )
    namespace = {
        "CPUParent": CPUParent,
        "torch": backend,
        "all_mla_use_flashmla": lambda model: model.use_flashmla,
        "is_deepseek_v41": lambda config: config.model_type == "deepseek_v41",
    }
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(runner_path), "exec", __future__.annotations.compiler_flag), namespace)
    return namespace["NPUModelRunner"], namespace["AscendDSparkSpeculator"], backend


class DeviceSnapshot:
    def __init__(self, values):
        self.values = values
        self.ready = False
        self.read_indices = []

    def __getitem__(self, index):
        if not self.ready:
            raise AssertionError("CPU read an unfinished D2H snapshot")
        self.read_indices.append(index)
        return self.values[index]


class TestMRV2HostLengthSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.update = staticmethod(load_length_consumer())

    def setUp(self):
        self.snapshot = DeviceSnapshot([999, 60, -999])
        self.runner = SimpleNamespace(
            _skip_spec_cpu_length_sync=False,
            speculator=object(),
            sync_spec_pp_cpu_counts=False,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3")),
            req_states=SimpleNamespace(
                req_id_to_index={"prompt": 0, "decode": 1, "new": 2},
                num_computed_tokens_cpu=[128, 64, 768],
            ),
            input_buffers=SimpleNamespace(seq_lens_cpu=[0, 0, 0]),
            num_computed_tokens_cpu=self.snapshot,
            num_computed_tokens_event=SimpleNamespace(synchronize=Mock(side_effect=self.complete_copy)),
        )
        self.output = SimpleNamespace(
            num_scheduled_tokens={"prompt": 16, "decode": 8, "new": 1},
            scheduled_cached_reqs=SimpleNamespace(req_ids=["prompt", "decode"]),
            scheduled_spec_decode_tokens={},
        )

    def complete_copy(self):
        self.snapshot.ready = True

    def test_cached_prompt_chunks_do_not_wait_or_read_unfinished_snapshot(self):
        self.update(self.runner, self.output, ["prompt"], [True])
        self.runner.num_computed_tokens_event.synchronize.assert_not_called()
        self.assertEqual(self.snapshot.read_indices, [])
        self.assertEqual(self.runner.input_buffers.seq_lens_cpu[0], 144)
        self.assertEqual(self.runner.req_states.num_computed_tokens_cpu[0], 128)

    def test_new_decode_request_preserves_prefix_count_from_reused_slot(self):
        self.update(self.runner, self.output, ["new"], [False])
        self.runner.num_computed_tokens_event.synchronize.assert_not_called()
        self.assertEqual(self.runner.input_buffers.seq_lens_cpu[0], 769)

    def test_mixed_batch_waits_only_for_continuing_decode_and_applies_rejections(self):
        self.update(self.runner, self.output, ["prompt", "decode", "new"], [True, False, False])
        self.runner.num_computed_tokens_event.synchronize.assert_called_once_with()
        self.assertEqual(self.snapshot.read_indices, [1])
        self.assertEqual(self.runner.req_states.num_computed_tokens_cpu, [128, 60, 768])
        self.assertEqual(self.runner.input_buffers.seq_lens_cpu, [144, 68, 769])

    def test_unscheduled_cached_requests_do_not_force_a_wait(self):
        self.update(self.runner, self.output, ["new"], [True])
        self.runner.num_computed_tokens_event.synchronize.assert_not_called()
        self.assertEqual(self.snapshot.read_indices, [])

    def test_device_length_path_keeps_optimistic_decode_bound(self):
        self.runner._skip_spec_cpu_length_sync = True
        self.update(self.runner, self.output, ["decode"], [False])
        self.runner.num_computed_tokens_event.synchronize.assert_not_called()
        self.assertEqual(self.runner.input_buffers.seq_lens_cpu[0], 72)
        self.assertEqual(self.runner.req_states.num_computed_tokens_cpu[1], 64)

    def test_missing_phase_information_keeps_conservative_wait(self):
        self.update(self.runner, self.output, ["decode"])
        self.runner.num_computed_tokens_event.synchronize.assert_called_once_with()
        self.assertEqual(self.runner.input_buffers.seq_lens_cpu[0], 68)

    def test_prefill_with_scheduled_drafts_still_needs_exact_snapshot(self):
        self.output.scheduled_spec_decode_tokens = {"decode": [1] * 7}
        self.update(self.runner, self.output, ["decode"], [True])
        self.runner.num_computed_tokens_event.synchronize.assert_called_once_with()
        self.assertEqual(self.runner.input_buffers.seq_lens_cpu[0], 68)

    def test_prefill_then_decode_consumes_snapshot_when_it_first_becomes_needed(self):
        self.update(self.runner, self.output, ["prompt"], [True])
        self.runner.num_computed_tokens_event.synchronize.assert_not_called()
        self.snapshot.values[0] = 140
        self.update(self.runner, self.output, ["prompt"], [False])
        self.runner.num_computed_tokens_event.synchronize.assert_called_once_with()
        self.assertEqual(self.runner.input_buffers.seq_lens_cpu[0], 156)

    def test_non_last_pp_rank_still_consumes_rejection_corrected_counts(self):
        self.runner.speculator = None
        self.runner.sync_spec_pp_cpu_counts = True
        self.update(self.runner, self.output, ["decode"], [False])
        self.runner.num_computed_tokens_event.synchronize.assert_called_once_with()
        self.assertEqual(self.runner.input_buffers.seq_lens_cpu[0], 68)


class TestMRV2TargetSnapshotPolicy(unittest.TestCase):
    def setUp(self):
        runner_cls, draft_cls, self.backend = load_snapshot_runner()
        self.runner = runner_cls()
        draft = draft_cls()
        draft.attn_architecture = "MLA"
        draft.use_dcp = False
        draft._draft_mla_uses_device_lengths = False
        self.runner.speculator = draft
        self.runner.model = SimpleNamespace(use_flashmla=True)
        self.runner.model_config = SimpleNamespace(
            architecture="KimiK3ForCausalLM", hf_config=SimpleNamespace(model_type="kimi_k3")
        )
        self.runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1, prefill_context_parallel_size=1)
        )
        self.runner.ascend_config = SimpleNamespace(
            sparse_kv_offload_config=SimpleNamespace(enabled=False), kvpp_config=SimpleNamespace(size=1)
        )
        self.runner.adaptive_verification = None
        self.runner.use_pp = False
        self.runner.sync_spec_pp_cpu_counts = False
        self.runner.parent_calls = []
        self.runner.req_states = SimpleNamespace(
            req_id_to_index={"prompt": 0, "decode": 1},
            num_computed_tokens_cpu=[128, 64],
            num_computed_tokens=SimpleNamespace(gpu=[128, 64]),
        )
        self.runner.input_buffers = SimpleNamespace(seq_lens_cpu=[0, 0])
        self.runner.num_computed_tokens_cpu = DeviceSnapshot([128, 60])
        self.runner.num_computed_tokens_event = SimpleNamespace(synchronize=Mock(), record=Mock())

    def test_target_and_draft_backend_matrix(self):
        for target_flashmla in (False, True):
            for draft_device_lengths in (False, True):
                for use_dcp in (False, True):
                    with self.subTest(target=target_flashmla, draft=draft_device_lengths, dcp=use_dcp):
                        self.runner.model.use_flashmla = target_flashmla
                        self.runner.speculator._draft_mla_uses_device_lengths = draft_device_lengths
                        self.runner.speculator.use_dcp = use_dcp
                        self.assertEqual(
                            self.runner._can_skip_spec_cpu_length_sync(),
                            target_flashmla and (draft_device_lengths or not use_dcp),
                        )

    def test_unaudited_or_uninitialized_draft_keeps_target_snapshot(self):
        for architecture in (None, "GQA", "SFA"):
            with self.subTest(architecture=architecture):
                self.runner.speculator.attn_architecture = architecture
                self.assertFalse(self.runner._can_skip_spec_cpu_length_sync())
        for draft in (None, object(), SimpleNamespace(_draft_mla_uses_device_lengths=False)):
            with self.subTest(draft=draft):
                self.runner.speculator = draft
                self.assertFalse(self.runner._can_skip_spec_cpu_length_sync())

    def test_explicit_unknown_or_required_consumer_cannot_bypass(self):
        for required in (None, True):
            with self.subTest(required=required):
                self.runner.speculator = SimpleNamespace(
                    requires_target_cpu_seq_lens=required, _draft_mla_uses_device_lengths=True
                )
                self.assertFalse(self.runner._can_skip_spec_cpu_length_sync())

    def test_existing_safety_guards_still_apply_to_independent_draft(self):
        guards = [
            (self.runner.model_config, "architecture", "OtherModel"),
            (self.runner.vllm_config.parallel_config, "pipeline_parallel_size", 2),
            (self.runner.vllm_config.parallel_config, "prefill_context_parallel_size", 2),
            (self.runner, "adaptive_verification", object()),
            (self.runner.ascend_config.sparse_kv_offload_config, "enabled", True),
            (self.runner.ascend_config.kvpp_config, "size", 2),
        ]
        for owner, name, blocked in guards:
            with self.subTest(guard=name):
                previous = getattr(owner, name)
                setattr(owner, name, blocked)
                self.assertFalse(self.runner._can_skip_spec_cpu_length_sync())
                setattr(owner, name, previous)

    def test_independent_draft_preserves_device_rejections_without_target_d2h_or_wait(self):
        self.runner._skip_spec_cpu_length_sync = self.runner._can_skip_spec_cpu_length_sync()
        self.assertTrue(self.runner._skip_spec_cpu_length_sync)
        output = SimpleNamespace(
            num_scheduled_tokens={"prompt": 16, "decode": 8},
            scheduled_cached_reqs=SimpleNamespace(req_ids=["prompt", "decode"]),
            scheduled_spec_decode_tokens={"decode": [1] * 7},
        )
        self.runner.postprocess_sampled([0, 1], None, [1, 3], [0, 4])
        self.runner._update_seq_lens_cpu(output, ["prompt", "decode"], [True, False])
        # Also verify the copy helper's defensive guard, independently of the
        # postprocess call-site guard.
        self.runner._copy_num_computed_tokens_to_cpu()
        self.assertEqual(self.runner.parent_calls, ["reject"])
        self.assertEqual(self.runner.req_states.num_computed_tokens.gpu, [128, 60])
        self.assertEqual(self.runner.req_states.num_computed_tokens_cpu, [128, 64])
        self.assertEqual(self.runner.input_buffers.seq_lens_cpu, [144, 72])
        self.backend.cuda.current_stream.assert_not_called()
        self.backend.npu.stream.assert_not_called()
        self.runner.num_computed_tokens_event.record.assert_not_called()
        self.runner.num_computed_tokens_event.synchronize.assert_not_called()
        self.assertEqual(self.runner.num_computed_tokens_cpu.read_indices, [])

    def test_prefill_device_count_advances_without_target_snapshot(self):
        self.runner._skip_spec_cpu_length_sync = self.runner._can_skip_spec_cpu_length_sync()
        self.runner.postprocess_num_computed_tokens(SimpleNamespace(idx_mapping=[0], num_scheduled_tokens=[16]))
        self.assertEqual(self.runner.req_states.num_computed_tokens.gpu, [144, 64])
        self.assertEqual(self.runner.parent_calls, ["advance"])
        self.backend.cuda.current_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
