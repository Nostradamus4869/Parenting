This is a Pytorch implementation of Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning

# Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning

More details of the paper and dataset will be released after it is published.


# The Code

## Requirements

Following is the suggested way to install the dependencies:

    pip install -r requirements.txt

Note that ``pytorch >=3.9``.

## Folder Structure

```tex
└── code-and-data
    ├── data										# Dataset examples, detailed version will be released soon
    ├── finetune_twoTask.py			# Gradient-based sensitivity and uncertainty calculation
    ├── finetune_weight.py			# Forward activation probability computation
    ├── ipt_analyse.py					# Unit type recognition
    ├── requirements.txt				# The python environment needed for Parenting
    ├── run_train_twoTask.sh		# Forward activation probability computation script
    ├── run_weights.sh					#	Key parameter mining script
    ├── run_tailored_tuning.sh  # Tailored tuning script
    ├── transformers						# Modified transformers code
    │   ├── modeling_llama.py
    │   ├── modeling_qwen2.py
    │   ├── trainer.py
    │   └── trainer_seq2seq.py
    └── utils										# Other anxiliary functions
        ├── lora_importance.py
        └── prompter.py
```

 ## Local Setup
  - First, create python environment based on `requirements.txt`
  - Then replace the corresponding files in the Transformers package with files in `transformers` folder, in which we modified the source code to adapt to our importance calculation method
  - To reproduce our method, excecute the following scripts in sequence.

```shell
bash run_weights.sh
bash run_train_twoTask.sh
bash run_tailored_tuning.sh
```

