This is a naive Pytorch implementation of Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning

# Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning

🎉 **Our paper has been accepted by ACL 2025 Main Conference!**

We are still working on optimizing the code in order to achieve a more convenient method for parameter space importance calculation.

Special thanks to the open-source assistance from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).


# The Code

## Requirements

Following is the suggested way to install the dependencies:

    pip install -r requirements.txt

Note that ``pytorch >=3.9``.

## Folder Structure

```tex
└── code-and-data
    ├── data										# Dataset examples
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
    - We are still working on **a convenient method** to calculate importance scores, based on `Class TrainerCallback` according to [ViT-benchmark](https://github.com/Artessay/ViT-Benchmark).
  - To reproduce our method, excecute the following scripts in sequence.

```shell
bash run_weights.sh
bash run_train_twoTask.sh
bash run_tailored_tuning.sh
```

