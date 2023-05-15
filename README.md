# Get Text Words Using Hugging Face Large Language Model (yanekyuk/bert-uncased-keyword-extractor)

BERT models are typically used by fine-tuning them for specific tasks, such as text classification, named entity recognition, or question answering. The pre-training process allows them to learn a general understanding of language, which can then be tailored to specific tasks with a relatively small amount of task-specific training data.  __bert-base-uncased__ is a pre-trained model that was introduced by Google as part of their BERT (Bidirectional Encoder Representations from Transformers) architecture. It's a popular model for a variety of natural language processing (NLP) tasks.  The 'base' in bert-base-uncased refers to the model size. BERT models come in two sizes: 'base' and 'large'. The 'base' model has 12 layers and 110 million parameters. The 'large' model, on the other hand, has 24 layers and 340 million parameters.  The 'base' model is smaller, faster, and requires less computational resources compared to the 'large' model.  The 'uncased' in __bert-base-uncased__ means that the text has been lower-cased before being fed into the model, and any information about the original casing (capitalization) of the words is lost. For example, 'The', 'THE', and 'the' would all be represented the same way in an uncased model.

The __yanekyuk/bert-uncased-keyword-extractor__ model is a fine-tuned version of __bert-base-uncased__, and it achieves the following results on the evaluation set.  It is fine tuned on the smaller of the BERT models and has very good performance when used as a function call looping over text or groups of text with multi-threading.  It also has very good performance when used with Spark User Defined Functions (UDFs).  The code is in a single notebook that can be run natively in Databricks or locally in VS CODE or your IDE of choice.  It has a environment built-in check to know if a hugging face model needs to be downloaded locally.

- Loss: 0.1247
- Precision: 0.8547
- Recall: 0.8825
- Accuracy: 0.9741
- F1: 0.8684


## Hugging Face Demonstration Highlighting Keywords and Keyphrases

![hf_example.png](/readme_images/hf_example.png)


### Clone Down the Repo into Databricks Workspace or Locally in VSCODE: <br>

- git clone https://github.com/robert-altmiller/hf_llms.git


### Step 1: Run the hf_keywords.py Notebook

![run_notebook.png](/readme_images/run_notebook.png)
