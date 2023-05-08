# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     DataCollatorForSeq2Seq,
# )
# from datasets import Dataset
# from torch.utils.data import DataLoader
# import torch
# from tqdm import tqdm
from base_generator import Seq2SeqBaseGenerator


def flatten_list(chunk_list):
    for chunk in chunk_list:
        if isinstance(chunk, list):
            yield from flatten_list(chunk)
        else:
            yield chunk


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def prompt_flant5(prompt_input: list, model_name='flan-t5-large', max_tokens=128,
                  clean=False, batch_size=16, verbose=False, **kwargs):

    _model_name_or_path = 'google/' + model_name if model_name.startswith('flan-t5') else model_name
    flant5 = Seq2SeqBaseGenerator(_model_name_or_path)

    outputs = flant5.generate(prompt_input,
                              num_return_sequences=kwargs.get('n', 1),
                              beam_size=kwargs.get('n', 1),
                              temperature=kwargs.get('temperature', 0),
                              max_new_tokens=max_tokens,
                              per_device_test_batch_size=batch_size,
                              verbose=verbose,)
    
    return flatten_list(outputs)

    # tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
    # model = AutoModelForSeq2SeqLM.from_pretrained(_model_name_or_path, device_map='auto')

    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=-100,
    #     pad_to_multiple_of=None,
    # )

    # if isinstance(prompt_input[0], str):
    #     prompt_input = [{'source': x} for x in prompt_input]
    
    # raw_datasets = Dataset.from_list(prompt_input, split='test')
    # column_names = raw_datasets.column_names
    
    # def preprocess_function(examples, prefix='', source_column='source'):
    #     padding = 'max_length'
    #     inputs = examples[source_column]
    #     inputs = [prefix + inp for inp in inputs]
    #     model_inputs = tokenizer(inputs, padding=padding, truncation=True)
    #     return model_inputs

    # processed_datasets = raw_datasets.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=1,
    #     remove_columns=column_names,
    #     desc="Running tokenizer on dataset",
    # )

    # dataloader = DataLoader(processed_datasets, collate_fn=data_collator, batch_size=batch_size)

    # predictions = []
    # for batch in tqdm(dataloader, total=len(dataloader), disable=not verbose,):
    #     with torch.no_grad():
    #         generated_output = model.generate(
    #             batch["input_ids"],
    #             attention_mask=batch["attention_mask"],
    #             num_beams=kwargs.get('n', 1),
    #             num_return_sequences=kwargs.get('n', 1),
    #             max_new_tokens=max_tokens,
    #             output_scores=True,
    #             return_dict_in_generate=True,
    #             temperature=kwargs.get('temperature', 0)
    #         )
    #         generated_tokens = generated_output.sequences
    #         if isinstance(generated_tokens, tuple):
    #             generated_tokens = generated_tokens[0]
    #         decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    #         preds = [pred.strip() for pred in decoded_preds]
    #         predictions += preds

    # predictions = list(map(lambda x: x.strip(), predictions))
    # # TODO: `[:len(test_dataset)]` is a hack to deal with the auto-filling of the last incomplete batch.
    # predictions = list(chunks(predictions, kwargs.get('n', 1)))[:len(processed_datasets)]

    # return flatten_list(predictions)


if __name__ == '__main__':
    res = prompt_flant5([{'source': 'are lions mammal?'}, {'source': 'can birds fly?'}], max_tokens=32)
    print(list(res))