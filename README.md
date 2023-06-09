# Say What You Mean! Large Language Models Speak Too Positively about Negative Commonsense Knowledge

This repo contains the experimental code and resources used in our ACL 2023 paper: [Say What You Mean! Large Language Models Speak Too Positively about Negative Commonsense Knowledge](https://github.com/jiangjiechen/uncommongen).


## Install Requirements

```bash
export PJ_HOME=${YOUR_WORKING_DIR}/uncommongen/
export OPENAI_API_KEY=${YOUR_API_KEY}

pip3 install -r requirements.txt
```


## Download Datasets

The CSK-PN dataset can be found at [[Google drive](https://drive.google.com/drive/folders/1KqeIUVhqh7rUqbJwuIykYKFOq-K3VuYw?usp=sharing)] in jsonline format.

Since running OpenAI models are costly, we also release the generated results by these LLMs along with the dataset (so it's a pretty big json per line). You will find them nested under `cg_pred`, `qa_pred`, etc.

## How to Run

> Note: OpenAI has deprecated `code-davinci-002`.

### Running constrained generation (CG)

For detailed parameters, please refer to `constrained_generation/llm_constrained_generation.py`.

An example: 

```bash
python3 constrained_generation/llm_constrained_generation.py -i ${INPUT_FILE} -o ${OUTPUT_FILE} -m ${MODEL_NAME} --posk ${POSK} --negk ${NEGK} -b 16 --cot none
```

### Running boolean question answering (QA)

For detailed parameters, please refer to `boolqa/llm_answer_prediction.py`.

An example:

```bash
python3 boolqa/llm_answer_prediction.py -i ${INPUT_FILE} -o ${OUTPUT_FILE} -m ${MODEL_NAME} --posk ${POSK} --negk ${NEGK} -b 16 --cot none
```


## Evaluation

### Evaluate constrained generation (CG)

```bash
python3 evaluation/eval_constrained_generation.py -i ${INPUT_FILE} -m ${MODEL_KEY}
```

Note that `${MODEL_KEY}` is the id of a generation in the input json file, typically in the form of `${MODEL_NAME}_ex-${POSK}p${NEGK}n`, such as `text-davinci-002_ex-3p3n`. Different parameters could result in different model keys. Please check the code carefully.

### Evaluate boolean question answering (QA)

```bash
python3 evaluation/eval_boolqa.py -i ${INPUT_FILE} -m ${MODEL_KEY}
```

Same note as the CG task.

## Citation

If you find our paper or resources useful, please kindly cite our paper. If you have any questions, please [contact us](mailto:jjchen19@fudan.edu.cn)!

```latex
@inproceedings{chen-etal-2023-say,
    title = "Say What You Mean! Large Language Models Speak Too Positively about Negative Commonsense Knowledge",
    author = "Chen, Jiangjie  and
      Shi, Wei  and
      Fu, Ziquan  and
      Cheng, Sijie  and
      Li, Lei  and
      Xiao, Yanghua",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.550",
    pages = "9890--9908",
    abstract = "Large language models (LLMs) have been widely studied for their ability to store and utilize positive knowledge. However, negative knowledge, such as {``}lions don{'}t live in the ocean{''}, is also ubiquitous in the world but rarely mentioned explicitly in text.What do LLMs know about negative knowledge?This work examines the ability of LLMs on negative commonsense knowledge.We design a constrained keywords-to-sentence generation task (CG) and a Boolean question answering task (QA) to probe LLMs.Our experiments reveal that LLMs frequently fail to generate valid sentences grounded in negative commonsense knowledge, yet they can correctly answer polar yes-or-no questions.We term this phenomenon the belief conflict of LLMs.Our further analysis shows that statistical shortcuts and negation reporting bias from language modeling pre-training cause this conflict.",
}
```
