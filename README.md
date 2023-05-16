# Get Text Words Using Hugging Face Large Language Model (yanekyuk/bert-uncased-keyword-extractor)

__BERT__ models are typically used by fine-tuning them for specific tasks, such as text classification, named entity recognition, or question answering. The pre-training process allows them to learn a general understanding of language, which can then be tailored to specific tasks with a relatively small amount of task-specific training data.  

__bert-base-uncased__ is a pre-trained model that was introduced by Google as part of their BERT (Bidirectional Encoder Representations from Transformers) architecture. It's a popular model for a variety of natural language processing (NLP) tasks.  The 'base' in bert-base-uncased refers to the model size. 

__BERT__ models come in two sizes: __base__ and __large__. The __base__ model has 12 layers and 110 million parameters. The __large__ model, on the other hand, has 24 layers and 340 million parameters.  The __base__ model is smaller, faster, and requires less computational resources compared to the __large__ model.  The 'uncased' in __bert-base-uncased__ means that the text has been lower-cased before being fed into the model.


## Hugging Face Demonstration Highlighting Keywords and Keyphrases

![hf_example.png](/readme_images/hf_example.png)

The __yanekyuk/bert-uncased-keyword-extractor__ model is a fine-tuned on the smaller version of the __BERT__ models: __bert-base-uncased__, and it achieves the following results on the evaluation set.

- Loss: 0.1247
- Precision: 0.8547
- Recall: 0.8825
- Accuracy: 0.9741
- F1: 0.8684


## Steps to Run Code to Get Keywords and Keyphrases <br>

### Step 1: Clone Down the Repo into Databricks Workspace or Locally in Visual Studio Code: <br>

- git clone https://github.com/robert-altmiller/hf_llms.git

### Step 2: Install the Libraries requirements.txt <br>

Requirements need to be installed prior to running the __hf_keywords.py__ file, and there is a requirements.txt in the local repository.  If you are using Databricks the __requirments.txt__ that is installed at the top of the __hf_keywords__ Python notebook.

- libraries that need to be installed are __pyspark__, __torch__, and __transformers__.

### Step 3: Run the hf_keywords.py Notebook <br>

The code is in a single notebook that can be run natively in Databricks, locally in VS Code or your development IDE of choice.  It has a environment built-in check for Databricks to run the keywords unit test with a Spark dataframe + UDF.<br>

It also has very good performance when used with Spark User Defined Functions (UDFs), looping over text in a Python list linearly, or groups of text using multi-threading.  Requirements need to be installed prior to running the notebook and there is a requirements.txt in the local repository that is called from the __hf_keywords__ Python notebook.

![run_notebook.png](/readme_images/run_notebook.png)

### Step 4: Analyzing the Results From the hf_keywords Notebook <br>

