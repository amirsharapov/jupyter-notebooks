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


PROMPT = '''\
You are an AI programming assistant, utilizing the Deepseek Coder model. You are capable of:

- Expertly answering questions related to programming and code
- Can parse code snippets and extract useful information

Your response should be a single JSON object with the following schema:

```json
{
    "functions": [
        {
            "name": {
                "type": "str",
                "description": "The name of the function"
            },
            "parameters": {
                "type": "array",
                "description": "A list of parameters for the function",
                "items": {
                    "type": "str",
                    "description": "The name of the parameter"
                }
            },
            "return_type": {
                "type": "str",
                "description": "The return type of the function"
            }
            "body": {
                "type": "str",
                "description": "The body of the function"
            }
        }
    ]
}
```

Example Prompt:
```
def hello_world():
    print("Hello, World!")


def add(a, b):
    return a + b
```

Example Response:
```json
{
    "functions": [
        {
            "name": "hello_world",
            "parameters": [],
            "return_type": "None",
        },
        {
            "name": "add",
            "parameters": ["a", "b"],
            "return_type": "int",
        }
    ]
}
```

### Instruction:
Given the following python code, please return a list of all the functions that are defined in the code.

```
{{code}}
```

### Response:
'''


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
        PROMPT.replace('{{code}}', open('main.py').read().replace('```', ''))
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
