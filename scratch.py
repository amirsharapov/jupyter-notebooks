import json
from pathlib import Path

from llama_cpp import Llama


ROOT_PATH = Path('/home/amir/.cache/lm-studio/models/')


class Model:
    path = [
        ROOT_PATH / 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_S.gguf',
        ROOT_PATH / 'TheBloke/deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_S.gguf',
        ROOT_PATH / 'QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf',
    ][0]

    instance = None
        
    @classmethod
    def get(cls):
        if not cls.instance:
            cls.instance = Llama(
                cls.path.as_posix(),
                n_gpu_layers=-1,
                n_ctx=1024 * 8,
                rope_freq_base=0,
                rope_freq_scale=0,
                n_batch=2048,
                verbose=False
            )

        return cls.instance

def get_prompt(name: str):
    if name == 'extract_python_functions':
        return open('assets/prompts/extract_python_functions.txt').read()
    
    if name == 'fix_json_syntax':
        return open('assets/prompts/fix_json_syntax.txt').read()
    
    raise ValueError(f'Prompt "{name}" not found')


def prettify(o: dict | list):
    return json.dumps(o, indent=2, ensure_ascii=False, default=str)


def add(a, b):
    return a + b


def parse_text_from_completion(response: dict):
    text = response['choices'][0]['text']
    text = text.strip().replace('```json', '').removesuffix('```').strip()
    return text


def parse_json_from_completion(response: dict):
    text = parse_text_from_completion(response)

    try:
        return json.loads(text)

    except json.JSONDecodeError as e:
        model = Model.get()
        response = model.create_completion(
            get_prompt('fix_json_syntax').replace('{{json}}', text).replace('{{error}}', str(e)),
            max_tokens=-1
        )

        return parse_json_from_completion(response)


def main():
    model = Model.get()

    response = model.create_completion(
        get_prompt('extract_python_functions').replace('{{code}}', open('main.py').read()),
        max_tokens=-1
    )

    text = parse_text_from_completion(response)

    try:
        json.loads(text)

    except json.JSONDecodeError as e:
        print('Response is not a valid JSON. Retrying...')
        
        response = model.create_completion(
            get_prompt('fix_json_syntax').replace('{{json}}', text).replace('{{error}}', str(e)),
            max_tokens=-1
        )

        text = parse_text_from_completion(response)

    print('Response Parsed')
    print('---')
    print(prettify(json.loads(text)))
    print('---')


if __name__ == '__main__':
    main()
