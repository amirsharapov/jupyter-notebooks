import json
from pathlib import Path

from llama_cpp import Llama


root = Path('/home/amir/.cache/lm-studio/models/TheBloke/')

MODELS = {
    'mistral-7b-instruct': root / 'Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_S.gguf',
    'deepseek-coder-6.7b-instruct': root / 'deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_S.gguf',
}


def prettify(o: dict | list):
    return json.dumps(o, indent=2, ensure_ascii=False, default=str)


PROMPT = open('assets/prompts/extract_python_functions.txt').read()


def main():
    model = Llama(
        MODELS['mistral-7b-instruct'].as_posix(),
        n_gpu_layers=-1,
        n_ctx=8192,
        rope_freq_base=0,
        rope_freq_scale=0,
        n_batch=512,
        verbose=False
    )
    
    response = model(
        PROMPT.replace('{{code}}', open('main.py').read())
    )

    print('Response:')
    print('---')
    print(prettify(response))
    print('---')

    print('Response Parsed')
    print('---')
    print(prettify(json.loads(response['choices'][0]['text'])))
    print('---')


if __name__ == '__main__':
    main()
