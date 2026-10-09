"""Verbatim lines of smoke-r2-2's vLLM v0.31.0 logs (r0_NV-a85ed6eb.log, r0_MX-9c69d830.log,
r0_NVv-32c0a109.log), for server.parse_server_log."""

NV_LINES = [
    (
        "(MainProcess pid=4136) INFO 10-05 02:13:04 [compilation.py:331] Enabled custom "
        "fusions: act_quant"
    ),
    (
        "(ApiServer_2 pid=4182) INFO 10-05 02:13:26 [compilation.py:331] Enabled custom "
        "fusions: act_quant"
    ),
    (
        "(ApiServer_3 pid=4183) INFO 10-05 02:13:27 [compilation.py:331] Enabled custom "
        "fusions: act_quant"
    ),
    (
        "(ApiServer_1 pid=4181) INFO 10-05 02:13:27 [compilation.py:331] Enabled custom "
        "fusions: act_quant"
    ),
    (
        "(ApiServer_0 pid=4180) INFO 10-05 02:13:27 [compilation.py:331] Enabled custom "
        "fusions: act_quant"
    ),
    (
        "(EngineCore pid=4179) INFO 10-05 02:13:30 [core.py:129] Initializing a V1 LLM "
        "engine (v0.31.0) with config: model='/local/qwen3-32b-nvfp4', "
        "speculative_config=None, tokenizer='/local/qwen3-32b-nvfp4', "
        "skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, "
        "tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, "
        "max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, "
        "pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, "
        "dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, "
        "quantization=compressed-tensors, quantization_config=None, enforce_eager=False, "
        "aux_output_config=AuxOutputConfig(enable_return_routed_experts=False, "
        "max_bytes=None), kv_cache_dtype=bfloat16, device_config=cuda, "
        "structured_outputs_config=StructuredOutputsConfig(backend='auto', "
        "disable_any_whitespace=False, disable_additional_properties=False, "
        "reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), "
        "observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, "
        "otlp_traces_endpoint=None, collect_detailed_traces=None, "
        "per_request_spec_decode_metrics='none', kv_cache_metrics=False, "
        "kv_cache_metrics_sample=0.01, cudagraph_metrics=False, "
        "enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, "
        "enable_mm_processor_stats=False, enable_logging_iteration_details=False, "
        "jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, "
        "served_model_name=fp4bench, enable_prefix_caching=False, "
        "enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': "
        "<CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', "
        "'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': "
        "['none'], 'ir_enable_torch_wrap': True, 'splitting_ops': "
        "['vllm::unified_attention_with_output', 'vllm::unified_mla_attention_with_output', "
        "'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', "
        "'vllm::qwen4_exp_ple_short_conv', 'vllm::qwen4_exp_qsa_with_output', "
        "'vllm::linear_attention', 'vllm::qwen_gdn_attention_core', "
        "'vllm::qwen_gdn_attention_core_fused_norm_packed', 'vllm::gdn_attention_core_xpu', "
        "'vllm::olmo_hybrid_gdn_full_forward', 'vllm::sparse_attn_indexer', "
        "'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::deepseek_v4_attention', "
        "'vllm::hpc_rope_norm_forward', 'vllm::unified_kv_cache_update', "
        "'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, "
        "'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], "
        "'encoder_cudagraph_max_vision_items_per_batch': 0, "
        "'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], "
        "'compile_ranges_endpoints': [16384], 'inductor_compile_config': "
        "{'enable_auto_functionalized_v2': False, 'combo_kernels': True, "
        "'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': "
        "<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, "
        "'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, "
        "104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, "
        "232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, "
        "464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, "
        "720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, "
        "976, 992, 1008, 1024], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': "
        "True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': "
        "False, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, "
        "'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, "
        "'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, "
        "'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, "
        "'max_cudagraph_capture_size': 1024, 'dynamic_shapes_config': {'type': "
        "<DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, "
        "'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': "
        "False, 'static_all_moe_layers': []}, "
        "kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], "
        "fused_add_rms_norm=['native'], gelu_and_mul_sparse=['triton', 'native']), "
        "enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, "
        "enable_jit_warmup=True, moe_backend='auto', sparse_indexer_topk_backend='auto', "
        "linear_backend='flashinfer_cutedsl', linear_backend_per_quant=None)"
    ),
    (
        "(EngineCore pid=4179) INFO 10-05 02:13:32 [__init__.py:1186] Using "
        "FlashInferCuteDslNvFp4LinearKernel for NVFP4 GEMM"
    ),
    (
        "(EngineCore pid=4179) INFO 10-05 02:13:33 [cuda.py:526] Using FLASHINFER attention "
        "backend out of potential backends: ['FLASHINFER', 'FLASH_ATTN', 'TRITON_ATTN', "
        "'FLEX_ATTENTION']."
    ),
    (
        "(EngineCore pid=4179) INFO 10-05 02:13:57 [flashinfer.py:903] FlashInfer resolved "
        "query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16, "
        "decode_backend=trtllm-gen, kv_cache_dtype=torch.bfloat16, arch=sm100"
    ),
    (
        "(EngineCore pid=4179) INFO 10-05 02:14:23 [gpu_worker.py:692] Available KV cache "
        "memory: 136.35 GiB"
    ),
    (
        "(EngineCore pid=4179) INFO 10-05 02:14:23 [kv_cache_utils.py:2464] GPU KV cache "
        "size: 558,496 tokens, Maximum concurrency for 4,096 tokens per request: 136.35x"
    ),
    (
        "(EngineCore pid=4179) INFO 10-05 02:14:58 [compilation.py:331] Enabled custom "
        "fusions: act_quant"
    ),
]

MX_LINES = [
    (
        "(EngineCore pid=102) INFO 10-05 02:03:44 [core.py:129] Initializing a V1 LLM engine "
        "(v0.31.0) with config: model='/local/qwen3-32b-mxfp4', speculative_config=None, "
        "tokenizer='/local/qwen3-32b-mxfp4', skip_tokenizer_init=False, tokenizer_mode=auto, "
        "revision=None, tokenizer_revision=None, trust_remote_code=False, "
        "dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=auto, "
        "tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, "
        "decode_context_parallel_size=1, dcp_comm_backend=ag_rs, "
        "disable_custom_all_reduce=False, quantization=compressed-tensors, "
        "quantization_config=None, enforce_eager=False, "
        "aux_output_config=AuxOutputConfig(enable_return_routed_experts=False, "
        "max_bytes=None), kv_cache_dtype=bfloat16, device_config=cuda, "
        "structured_outputs_config=StructuredOutputsConfig(backend='auto', "
        "disable_any_whitespace=False, disable_additional_properties=False, "
        "reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), "
        "observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, "
        "otlp_traces_endpoint=None, collect_detailed_traces=None, "
        "per_request_spec_decode_metrics='none', kv_cache_metrics=False, "
        "kv_cache_metrics_sample=0.01, cudagraph_metrics=False, "
        "enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, "
        "enable_mm_processor_stats=False, enable_logging_iteration_details=False, "
        "jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, "
        "served_model_name=fp4bench, enable_prefix_caching=False, "
        "enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': "
        "<CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', "
        "'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': "
        "['none'], 'ir_enable_torch_wrap': True, 'splitting_ops': "
        "['vllm::unified_attention_with_output', 'vllm::unified_mla_attention_with_output', "
        "'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', "
        "'vllm::qwen4_exp_ple_short_conv', 'vllm::qwen4_exp_qsa_with_output', "
        "'vllm::linear_attention', 'vllm::qwen_gdn_attention_core', "
        "'vllm::qwen_gdn_attention_core_fused_norm_packed', 'vllm::gdn_attention_core_xpu', "
        "'vllm::olmo_hybrid_gdn_full_forward', 'vllm::sparse_attn_indexer', "
        "'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::deepseek_v4_attention', "
        "'vllm::hpc_rope_norm_forward', 'vllm::unified_kv_cache_update', "
        "'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, "
        "'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], "
        "'encoder_cudagraph_max_vision_items_per_batch': 0, "
        "'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], "
        "'compile_ranges_endpoints': [16384], 'inductor_compile_config': "
        "{'enable_auto_functionalized_v2': False, 'combo_kernels': True, "
        "'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': "
        "<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, "
        "'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, "
        "104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, "
        "232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, "
        "464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, "
        "720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, "
        "976, 992, 1008, 1024], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': "
        "True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': "
        "False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'enable_sp': False, "
        "'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, "
        "'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, "
        "'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, "
        "'max_cudagraph_capture_size': 1024, 'dynamic_shapes_config': {'type': "
        "<DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, "
        "'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': "
        "False, 'static_all_moe_layers': []}, "
        "kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], "
        "fused_add_rms_norm=['native'], gelu_and_mul_sparse=['triton', 'native']), "
        "enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, "
        "enable_jit_warmup=True, moe_backend='auto', sparse_indexer_topk_backend='auto', "
        "linear_backend='flashinfer_cutedsl', linear_backend_per_quant=None)"
    ),
    (
        "(EngineCore pid=102) INFO 10-05 02:03:49 [__init__.py:975] Using "
        "FlashInferMxFp4LinearKernel for MXFP4 GEMM"
    ),
    (
        "(EngineCore pid=102) INFO 10-05 02:03:49 [cuda.py:526] Using FLASHINFER attention "
        "backend out of potential backends: ['FLASHINFER', 'FLASH_ATTN', 'TRITON_ATTN', "
        "'FLEX_ATTENTION']."
    ),
    (
        "(EngineCore pid=102) INFO 10-05 02:04:20 [flashinfer.py:903] FlashInfer resolved "
        "query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16, "
        "decode_backend=trtllm-gen, kv_cache_dtype=torch.bfloat16, arch=sm100"
    ),
    (
        "(EngineCore pid=102) INFO 10-05 02:04:44 [gpu_worker.py:692] Available KV cache "
        "memory: 136.79 GiB"
    ),
    (
        "(EngineCore pid=102) INFO 10-05 02:04:44 [kv_cache_utils.py:2464] GPU KV cache "
        "size: 560,288 tokens, Maximum concurrency for 4,096 tokens per request: 136.79x"
    ),
]

NV_LOG = "\n".join(NV_LINES) + "\n"
MX_LOG = "\n".join(MX_LINES) + "\n"
NV_ENGINE_CONFIG = NV_LINES[5]
MX_ENGINE_CONFIG = MX_LINES[0]

NV_AOT_LOAD_LINE = (
    "(EngineCore pid=4179) INFO 10-05 02:13:54 [decorators.py:312] Directly load AOT compilation "
    "from path /root/.cache/vllm/torch_compile_cache/torch_aot_compile/"
    "6bbb8ec68251b3dd50f52779e4834d877f6cad4b619e05abbfaf230a53454814/rank_0_0/model"
)
MX_AOT_LOAD_LINE = (
    "(EngineCore pid=102) INFO 10-05 02:04:15 [decorators.py:312] Directly load AOT compilation "
    "from path /root/.cache/vllm/torch_compile_cache/torch_aot_compile/"
    "ee195ad7357b0b848cff3fc7454b8791a7a1280e23029e21cdf0e2911f949499/rank_0_0/model"
)
NVV_CACHE_DIR_LINE = (
    "(EngineCore pid=8130) INFO 10-05 02:49:30 [backends.py:1090] Using cache directory: "
    "/root/.cache/vllm/torch_compile_cache/7e47068371/rank_0_0/backbone for vLLM's torch.compile"
)
NVV_AOT_SAVE_LINE = (
    "(EngineCore pid=8130) INFO 10-05 02:50:03 [decorators.py:717] saved AOT compiled function to "
    "/root/.cache/vllm/torch_compile_cache/torch_aot_compile/"
    "985c0e220d0049d592e211fd0a3b75c09b1b3df89ee3bdfa63b93290bac65942/rank_0_0/model"
)
NV_AOT_HASH = "6bbb8ec68251b3dd50f52779e4834d877f6cad4b619e05abbfaf230a53454814"
MX_AOT_HASH = "ee195ad7357b0b848cff3fc7454b8791a7a1280e23029e21cdf0e2911f949499"
NVV_CACHE_DIR_HASH = "7e47068371"
NVV_AOT_HASH = "985c0e220d0049d592e211fd0a3b75c09b1b3df89ee3bdfa63b93290bac65942"
