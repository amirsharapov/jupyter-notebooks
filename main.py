import json

from src import llms
from src.common import prettify
from src.paths import PROMPTS_DIRECTORY


def main():
    prompts_base_path = PROMPTS_DIRECTORY / 'extract_python_functions'
    prompts_paths = {
        'system': prompts_base_path / 'system.txt',
        'user': prompts_base_path / 'user.txt'
    }

    for model in [
        # 'llama3',
        'mistral',
        # 'deepseek-coder'
    ]:
        response = llms.create_chat_completion(
            model,
            prompts_paths['system'].read_text(),
            prompts_paths['user'].read_text().replace('{{ code }}', open('main.py').read()),
            verbose=True,
            output_format='json'
        )
        
        # response = llama3.create_chat_completion(
        #     prompts_paths['system'].read_text(),
        #     prompts_paths['user'].read_text().replace('{{ code }}', open('scratch_1.py').read()),
        #     verbose=True
        # )

        print(prettify(response))
        print(prettify(json.loads(response['choices'][0]['text'])))

    # response = mistral.create_chat_completion(
    #     prompts_paths['system'].read_text(),
    #     prompts_paths['user'].read_text().replace('{{ code }}', open('scratch_1.py').read()),
    #     verbose=True
    # )

    # print(prettify(response))
    # print(response['choices'][0]['text'])


if __name__ == '__main__':
    main()
