import time
from typing import Literal
from pathlib import Path

from llama_cpp import Llama, LlamaGrammar

from src import paths
from src.common import prettify


def list_lm_studio_gguf_models() -> list[Path]:
    return list(paths.MODELS_DIRECTORY.glob('**/*.gguf'))


def load_lm_studio_gguf_model_lookup(save_to_disk: bool = True) -> dict[str, Path]:
    models = list_lm_studio_gguf_models()
    models = sorted(models, key=lambda m: m.stem)

    lookup = {
        (model.parent.parent.name + '/' + model.stem): model.as_posix() for
        model in
        models
    }

    if save_to_disk:
        Path('models.json').write_text(prettify(lookup))

    return lookup


MODEL_PARAMS = {
    'llama-2-7b-chat.Q3_K_L': {
        'stop_words': [
            '<|eot_id|>'
        ]
    }
}


def create_chat_completion(
    model: str | Path,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    verbose: bool = False,
    grammar: str | Path = None
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
        grammar=grammar
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
    model: str | Path,
    system_message: str,
    user_message: str,
    temperature: float = 0.1,
    verbose: bool = False,
    grammar: str | Path = None
):
    model_paths = load_lm_studio_gguf_model_lookup()
    model = model_paths[model] if isinstance(model, str) else model
    model = Path(model).as_posix()

    grammar=Path(grammar) if grammar else None

    llm = Llama(
        model,
        n_ctx=1024 * 16,
        n_gpu_layers=-1,
        verbose=verbose,
        temperature=temperature
    )

    grammar = LlamaGrammar.from_file(grammar)

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
            if delta['content'] in MODEL_PARAMS.get(model, {}).get('stop_words', []):
                break
            
            yield chunk
