import json
from pathlib import Path

from src import llms
from src.common import prettify
from src.paths import PROMPTS_DIRECTORY


def main():
    task_assets = Path('./tasks/identify_python_functions')

    prompts_paths = {
        'system': task_assets / 'prompts/system.txt',
        'user': task_assets / 'prompts/user.txt'
    }

    for model in [
        'deepseek-coder-7b-instruct-v1.5-Q5_K_M'
    ]:
        response = llms.create_chat_completion(
            model,
            prompts_paths['system'].read_text(),
            prompts_paths['user'].read_text().replace('{{ code }}', open('main.py').read()),
            verbose=True,
            grammar=task_assets / 'grammars/output.gbnf'
        )
        
        # response = llama3.create_chat_completion(
        #     prompts_paths['system'].read_text(),
        #     prompts_paths['user'].read_text().replace('{{ code }}', open('scratch_1.py').read()),
        #     verbose=True
        # )

        print(prettify(response))
        print(prettify(json.loads(response['choices'][0]['text'])))
        print(json.loads(response['choices'][0]['text'])['function_definitions'][0]['body'])

    # response = mistral.create_chat_completion(
    #     prompts_paths['system'].read_text(),
    #     prompts_paths['user'].read_text().replace('{{ code }}', open('scratch_1.py').read()),
    #     verbose=True
    # )

    # print(prettify(response))
    # print(response['choices'][0]['text'])


if __name__ == '__main__':
    main()
