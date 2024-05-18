# Advancing Multi-Tool Usage with Chain-of-Abstraction Reasoning

## Objective

### Problem Statement
One major issue we observe in current Large Language Models (LLMs) is their tendency to hallucinate, particularly when dealing with factual information. Despite being trained on vast amounts of data, LLMs essentially perform next-word prediction, leading to inaccuracies. Moreover, LLMs struggle with mathematical problem-solving.

Our proposal is to fine-tune LLMs for Chain-of-Thought reasoning by providing them with questions and enabling them to determine the appropriate tools needed to answer correctly. We aim to fine-tune LLMs to execute not only single-tool operations but also multiple-tool operations, such as conducting a Wikipedia search followed by an arithmetic operation for a given question.

Our study is mainly based on https://arxiv.org/pdf/2401.17464 paper.


## Highlights
- **Enhanced Data Filtering:** While the original CoA paper managed to filter only 15% of the synthetic data from LLaMA-70B, our custom Filtering Module achieved a much higher filtration rate of approximately 50%. This significant improvement highlights the efficacy of our data processing approach.
- **Unified Model Training:** Unlike the CoA paper, which trained separate models on different datasets, we successfully trained a single model on all datasets. This unified approach not only taught the model to use all tools effectively but also allowed us to experiment with multi-tool chaining, whereas the COA paper employed single tool chaining. This distinction underscores the versatility and robustness of our model.

## Dataset Generation
All the scripts for prompting the Gemini and Llama models can be found in the following files:
- **Math tool:** `Filtering Module/math.ipynb`
- **Wiki tool:** `Data Generation/hotpot_qa.ipynb` and `Data Generation/wiki.ipynb`
- **CommonSense multitool:** `Data Generation/commonsense_data_generation.ipynb`
- **Synthetic multitool:** `Data Generation/hotpot_qa_syn.ipynb`

## Tool Parsing and Filtering Modules
- **Math tool:** `Filtering Module/math.ipynb`
- **Wiki tool:** `Filtering Module/wiki_final.ipynb`
- **CommonSense multitool:** `Filtering Module/Multitool3.ipynb`
- **Synthetic multitool:** `Filtering Module/Multitool_syn.ipynb`

### Final Data Formatting
Making final Q and C pairs before training:
- `final_data_format.ipynb`

## Model Training Code
- **Old Prompt Training:** `Model_training/train_gemma.py` and `Model_training/train_mistral.py`
  - These files contain the training prompts for both models and respective training sessions, which led to very hallucinated outputs.

- **Final Prompt Training:** `Model_training/train_gemma2.py` and `Model_training/train_mistral2.py`

## Model Inferencing
- `model_testing.ipynb`
- `vllm_trained_model_testing.ipynb`

### Tool Parsing after Inference
Using the same scripts as filtering on new paths:
- `model_testing_tools.ipynb`
- `Filtering Module/math.ipynb`
- `Filtering Module/Multitool3.ipynb`
- `Filtering Module/Multitool_syn.ipynb`

## Other
- **Similarity score helper:** `similarity_check.ipynb`
- **Tool helpers exploration:** `langchain_tool_explore.ipynb`

## Wandb Dashboard
- [Final Mistral Training Dashboard](https://wandb.ai/anushkayadav/huggingface/reports/Final-Mistral-traininbg--Vmlldzo3OTk4MTQ1)

## Future Work
- Training separate models for each dataset and checking accuracy
- Generating more data for training
- Using better teacher models like GPT-4 or LLaMA-70B for synthetic data
