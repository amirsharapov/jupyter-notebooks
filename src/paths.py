from pathlib import Path

MODELS_DIRECTORY = Path('/home/amir/.cache/lm-studio/models/')
PROMPTS_DIRECTORY = Path('./assets/prompts')

MISTRAL_7B_INSTRUCT_MODEL_PATH = MODELS_DIRECTORY / 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_S.gguf'
DEEPSEEK_CODER_6_7B_MODEL_PATH = MODELS_DIRECTORY / 'TheBloke/deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_S.gguf'
LLAMA_3_8B_INSTRUCT_MODEL_PATH = MODELS_DIRECTORY / 'lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf'

JSON_GRAMMAR_GBNF_PATH = Path('./assets/grammars/generic/json.gbnf')
JSON_ARR_GRAMMAR_GBNF_PATH = Path('./assets/grammars/generic/json_arr.gbnf')

EXTRACT_PYTHON_FUNCTIONS_OUTPUT_GRAMMAR = Path('./assets/grammars/task_outputs/extract_python_functions.gbnf')