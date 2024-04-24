from transformers import AutoModel, AutoTokenizer


def main():
    model_path = 'QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf'
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    input_text = 'Hello. How are you?'
    input_ids = tokenizer(input_text, return_tensors='pt')

    output = []

    next_token = model.generate(
        input_ids=input_ids,
        max_length=1,
        num_return_sequences=1,
        streaming=True,
    )

    while next_token != tokenizer.eos_token_id:
        output.append(next_token)
        next_token = model.generate(
            input_ids=next_token,
            max_length=1,
            num_return_sequences=1,
            streaming=True,
        )

    output_text = tokenizer.decode(output)

    print(output_text)


if __name__ == '__main__':
    main()
