import os

# METHODOLOGY.md#stack
VLLM_IMAGE = (
    "vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b"
)
VLLM_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
EXPECTED_VERSIONS = {
    "vllm": "0.31.0",
    "flashinfer-python": "0.7.0.post1",
    "flashinfer-cubin": "0.7.0.post1",
    "torch": "2.13.0",
}
LLMCOMPRESSOR_VERSION = "0.14.0"
LIB_REPO = "https://github.com/local-inference-lab/llm-inference-bench"
LIB_COMMIT = "c71ec1f2b34a4f1c8f702f1750ccd70da0e389e2"
LIB_DIR = "/opt/llm-inference-bench"

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

DATA_DIR_ENV, LOCAL_DIR_ENV = "FP4BENCH_DATA_DIR", "FP4BENCH_LOCAL_DIR"
CONTAINER_DATA_DIR = "/data"
HF_CACHE = "/root/.cache/huggingface"
VLLM_CACHE = "/root/.cache/vllm"
DATA_DIR = os.environ.get(DATA_DIR_ENV, CONTAINER_DATA_DIR)
RESULTS_DIR = "/results"
LOCAL_DIR = os.environ.get(LOCAL_DIR_ENV, "/local")
NV_PROVENANCE_FILE = "fp4bench_quant.json"
SHAREGPT_PATH = f"{DATA_DIR}/{SHAREGPT_FILE}"
M1_PROMPTS_PATH = f"{DATA_DIR}/m1_prompts.json"
C_PROMPTS_PATH = f"{DATA_DIR}/c_prompts.json"
NLL_PROMPTS_PATH = f"{DATA_DIR}/nll_prompts.json"
BF16_REF_NLL_PATH = f"{DATA_DIR}/bf16_reference_nll.json"

SERVED_NAME = "fp4bench"
HOST, PORT = "127.0.0.1", 8000
BASE_URL = f"http://{HOST}:{PORT}"

SEED = 0
M1_INPUT_LEN = 1024
M1_N1, M1_N2 = 128, 1152
M1_SETS, M1_SET_SIZE = 6, 512
M2_MAX_TOKENS = 2048
NLL_PROMPTS, NLL_LEN = 64, 512
NLL_MAX_DEGRADATION = 0.5

DELTA = 0.02
CI_LEVEL = 0.90
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_LEVEL = 0.95
M1_MEAN_CONTEXT = M1_INPUT_LEN + (M1_N1 + M1_N2) // 2
HBM_PEAK_BYTES_S = 8e12  # METHODOLOGY.md#bytes-model
