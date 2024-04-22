import json
from pathlib import Path

from src import llms
from src.common import prettify


def main():
    task_assets = Path('./tasks/identify_python_functions')

    prompts_paths = {
        'system': task_assets / 'prompts/system.txt',
        'user': task_assets / 'prompts/user.txt'
    }

    for model in [
        'lmstudio-community/Meta-Llama-3-8B-Instruct-Q5_K_M',
        'LoneStriker/deepseek-coder-7b-instruct-v1.5-Q5_K_M'
    ]:
        response = llms.create_chat_completion(
            model,
            prompts_paths['system'].read_text(),
            prompts_paths['user'].read_text().replace('{{ code }}', open('scratch_1.py').read()),
            verbose=True,
            grammar=task_assets / 'grammars/output.gbnf'
        )

        print(prettify(response))
        print(prettify(json.loads(response['choices'][0]['text'])))


if __name__ == '__main__':
    main()
