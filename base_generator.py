# -*- coding: utf-8 -*-

import sys
import os
import torch
import ujson as json
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


accelerator = Accelerator()
torch.cuda.manual_seed_all(42)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def save_jsonl(jsl, output_file):
    with open(output_file, 'w') as f:
        for js in jsl:
            f.write(json.dumps(js, ensure_ascii=False) + '\n')
    return output_file


class Seq2SeqBaseGenerator:
    def __init__(self, model_name_or_path, cache_dir=None, use_auth_token=False, **model_kwargs):
        # turn on deepspeed if the model is too large
        max_retry_times = 3
        for i in range(max_retry_times):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_dir if cache_dir else None,
                    use_auth_token=use_auth_token
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_dir if cache_dir else None,
                    use_auth_token=use_auth_token,
                    **model_kwargs
                )
                break
            except Exception as e:
                print(e)
                print(f'* Reaching huggingface: {i}/{max_retry_times}')
        
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

    def preprocess_function(self, examples, max_source_length):
        padding = 'max_length'
        inputs = examples[self.source_column]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        return model_inputs
    
    def make_source_from_list(self, data):
        return [{'source': x} for x in data]

    def generate(self,
                 test_data_or_file,
                 output_file=None,
                 prefix='',
                 source_column='source',
                 beam_size=1,
                 num_return_sequences=1,
                 max_source_length=None,
                 per_device_test_batch_size=4,
                 verbose=True,
                 num_proc=1,
                 **generate_kwargs):
        assert num_return_sequences <= beam_size

        self.source_column = source_column
        self.prefix = prefix

        if isinstance(test_data_or_file, list):
            if isinstance(test_data_or_file[0], str):
                test_data_or_file = self.make_source_from_list(test_data_or_file)
            raw_datasets = Dataset.from_list(test_data_or_file, split='test')
        else:   # file
            extension = test_data_or_file.split(".")[-1]
            raw_datasets = load_dataset(extension, data_files={'test': test_data_or_file})['test']
        column_names = raw_datasets.column_names

        with accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                self.preprocess_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=column_names,
                desc="Running tokenizer on dataset",
                fn_kwargs={
                    'max_source_length': max_source_length,
                }
            )
        # test_dataset = processed_datasets['test']
        test_dataset = processed_datasets
        test_dataloader = DataLoader(test_dataset, collate_fn=self.data_collator, batch_size=per_device_test_batch_size)
        
        self.model, test_dataloader = accelerator.prepare(self.model, test_dataloader)

        predictions = []
        scores = []
        for batch in tqdm(test_dataloader, total=len(test_dataloader),
                          disable=not verbose and not accelerator.is_local_main_process,
                          desc=prefix):
            with torch.no_grad():
                generated_output = accelerator.unwrap_model(self.model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=beam_size,
                    num_return_sequences=num_return_sequences,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generate_kwargs
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_output.sequences, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                preds = [pred.strip() for pred in decoded_preds]
                predictions += preds

                # gather scores
                # generated_scores = accelerator.gather(generated_output.scores)
                # generated_scores = (x.cpu().numpy() for x in
                #                     generated_scores)  # before softmax: (max_length-1, [bsz * beam_size, vocab])
                # scores += generated_scores

        predictions = list(map(lambda x: x.strip(), predictions))
        # TODO: `[:len(test_dataset)]` is a hack to deal with the auto-filling of the last incomplete batch.
        predictions = list(chunks(predictions, num_return_sequences))[:len(test_dataset)] 
        if output_file is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                with open(output_file, 'w') as f:
                    for pred in predictions:
                        f.write(' '.join(pred) + '\n')

        return predictions


def test():
    input_data = [
                     {'source': '1. is time changing globally? (A) yes (B) no', 'answer': 'time'},
                     {'source': '2. is rabbit the largest animal on earth?', 'answer': 'one thing'},
                     {'source': '3. is donald trump american? \n (A) yes (B) no', 'answer': 'two birds'},
                     {'source': '4. is time changing globally? (A) yes (B) no', 'answer': 'time'},
                     {'source': '5. is rabbit the largest animal on earth?', 'answer': 'one thing'},
                     {'source': '6. is donald trump american? \n (A) yes (B) no', 'answer': 'two birds'},
                     {'source': '7. is time changing globally? (A) yes (B) no', 'answer': 'time'},
                     {'source': '8. is rabbit the largest animal on earth?', 'answer': 'one thing'},
                     {'source': '9. is donald trump american? \n (A) yes (B) no', 'answer': 'two birds'},
                    #  {'source': '4. who is the president of america?', 'answer': 'three values'},
                 ]

    generator = Seq2SeqBaseGenerator('t5-small')
    results = generator.generate(input_data, beam_size=2, num_proc=1, prefix='translate: ')

    if accelerator.is_main_process:
        print(results)


if __name__ == '__main__':
    test()
