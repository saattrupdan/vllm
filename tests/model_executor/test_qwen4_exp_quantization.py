# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest


@pytest.mark.parametrize("platform", ["nvidia", "amd"])
def test_qwen4_exp_lm_head_receives_target_quant_config(platform: str) -> None:
    module_path = f"vllm.models.qwen4_exp.{platform}.model"
    module = __import__(module_path, fromlist=["Qwen4ExpForCausalLM"])
    model_cls = module.Qwen4ExpForCausalLM
    quant_config = Mock()
    vllm_config = _make_vllm_config(quant_config=quant_config)

    with (
        patch(f"{module_path}.Qwen4ExpModel") as mock_model,
        patch(f"{module_path}.ParallelLMHead") as mock_lm_head,
        patch(f"{module_path}.LogitsProcessor"),
        patch(f"{module_path}.enable_qwen4_exp_low_latency_gemm"),
        patch.object(model_cls, "set_moe_parameters"),
    ):
        mock_model.return_value.make_empty_intermediate_tensors = Mock()
        mock_model.return_value.layers = []
        model_cls(vllm_config=vllm_config)

    assert mock_lm_head.call_args.kwargs["quant_config"] is quant_config


@pytest.mark.parametrize("platform", ["nvidia", "amd"])
def test_qwen4_exp_mtp_lm_head_receives_draft_quant_config(platform: str) -> None:
    module_path = f"vllm.models.qwen4_exp.{platform}.mtp"
    module = __import__(module_path, fromlist=["Qwen4ExpMTP"])
    model_cls = module.Qwen4ExpMTP
    target_quant_config = Mock()
    draft_quant_config = Mock()
    vllm_config = _make_vllm_config(quant_config=target_quant_config)
    pp_group = Mock(is_last_rank=True)

    with (
        patch(f"{module_path}.Qwen4ExpMultiTokenPredictor") as mock_model,
        patch(f"{module_path}.ParallelLMHead") as mock_lm_head,
        patch(f"{module_path}.LogitsProcessor"),
        patch(f"{module_path}.get_draft_quant_config", return_value=draft_quant_config),
        patch(f"{module_path}.get_pp_group", return_value=pp_group),
        patch(f"{module_path}.enable_qwen4_exp_low_latency_gemm"),
        patch.object(model_cls, "set_moe_parameters"),
    ):
        mock_model.return_value.layers = []
        model_cls(vllm_config=vllm_config)

    assert mock_lm_head.call_args.kwargs["quant_config"] is draft_quant_config


def _make_vllm_config(quant_config: Mock) -> Mock:
    hf_config = Mock()
    hf_config.tie_word_embeddings = False
    hf_config.vocab_size = 128
    hf_config.hidden_size = 64

    vllm_config = Mock()
    vllm_config.model_config.hf_text_config = hf_config
    vllm_config.model_config.dtype = "bfloat16"
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.scheduler_config = Mock()
    vllm_config.quant_config = quant_config
    return vllm_config
