import time
from typing import Literal

from llama_cpp import Llama, LlamaGrammar

from src import paths


ModelLiteral = Literal[
    'llama3',
    'mistral',
    'deepseek-coder'
]

OutputFormatLiteral = Literal[
    'json',
    'json-arr',
    'plain'
]

MODEL_PATHS = {
    'llama3': paths.LLAMA_3_8B_INSTRUCT_MODEL_PATH,
    'mistral': paths.MISTRAL_7B_INSTRUCT_MODEL_PATH,
    'deepseek-coder': paths.DEEPSEEK_CODER_6_7B_MODEL_PATH
}

CUSTOM_STOPWORDS = {
    'llama3': [
        '<|eot_id|>',
    ],
}


def create_chat_completion(
    model: ModelLiteral,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    verbose: bool = False,
    output_format: OutputFormatLiteral = None
):
    response = {
        'prompts': [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ],
        'choices': [
            {
                'text': ''
            }
        ],
        'usage': {
            'output_tokens': 0
        }
    }

    output = stream_chat_completion(
        model=model,
        system_message=system_prompt,
        user_message=user_prompt,
        temperature=temperature,
        verbose=verbose,
        output_format=output_format
    )

    start = time.time()

    for i, token in enumerate(output):
        if i == 0:
            response['usage']['time_since_first_token'] = time.time() - start

        if 'content' in token['choices'][0]['delta']:
            response['usage']['output_tokens'] += 1
            response['choices'][0]['text'] += token['choices'][0]['delta']['content']
        
        if verbose:
            print(f'Generated tokens: {response["usage"]["output_tokens"]} / ?', end='\r')

    if verbose:
        print('')

    response['usage']['time_since_last_token'] = time.time() - start
    
    return response


def stream_chat_completion(
    model: ModelLiteral,
    system_message: str,
    user_message: str,
    temperature: float = 0.1,
    verbose: bool = False,
    output_format: OutputFormatLiteral = 'plain'
):
    llm = Llama(
        model_path=MODEL_PATHS[model].as_posix(),
        n_ctx=1024 * 16,
        n_gpu_layers=-1,
        verbose=verbose,
        temperature=temperature
    )

    # grammar = None

    # if output_format == 'json':
    #     grammar = LlamaGrammar.from_file(paths.JSON_GRAMMAR_GBNF_PATH)
    # if output_format == 'json-arr':
    #     grammar = LlamaGrammar.from_file(paths.JSON_ARR_GRAMMAR_GBNF_PATH)

    grammar = LlamaGrammar.from_file(paths.EXTRACT_PYTHON_FUNCTIONS_OUTPUT_GRAMMAR)

    output = llm.create_chat_completion(
        messages=[
            {
                'role': 'system',
                'content': system_message
            },
            {
                'role': 'user',
                'content': user_message
            }
        ],
        stream=True,
        temperature=temperature,
        grammar=grammar
    )
    
    for chunk in output:
        delta = chunk['choices'][0]['delta']
        
        if 'content' in delta:
            if delta['content'] in CUSTOM_STOPWORDS.get(model, []):
                break
            
            yield chunk
