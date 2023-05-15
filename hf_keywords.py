# Databricks notebook source
# DBTITLE 1,Install Libraries - Transformers and Pyspark
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

# DBTITLE 1,Library Imports
import string
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,Modify Python String With Edits and Changes
def modify_python_str(
        inputstr: str, 
        lowercase: bool = False,
        uppercase: bool = False,
        removenumbers: bool = False,
        removespaces: bool = False,
        removepunctuation: bool = False,
        singledashes: bool = False,
        periodforunderscore: bool = False
    )-> string:
    """modified a python string with edits and changes"""
    if lowercase: inputstr = inputstr.lower()
    if uppercase: inputstr = inputstr.upper()
    if periodforunderscore: inputstr = inputstr.replace(".", "_")
    if removenumbers: inputstr = re.sub(r'[0-9]', '', inputstr)
    if removespaces: inputstr = inputstr.replace(' ', '')
    if removepunctuation:
        punctuation = [punct for punct in str(string.punctuation)]
        punctuation.remove("-")
        for punct in punctuation:
            inputstr = inputstr.replace(punct, '')
    if singledashes: inputstr = re.sub(r'(-)+', r'-', inputstr)
    return inputstr

# COMMAND ----------

# DBTITLE 1,Get Hugging Face Keywords
# initial tokenizer and model
hf_model_name = "yanekyuk/bert-uncased-keyword-extractor"
print(f"initializing hf model: {hf_model_name}.....\n")
tokenizer_keyword = AutoTokenizer.from_pretrained(hf_model_name)
model_keyword = AutoModelForTokenClassification.from_pretrained(hf_model_name)
ner_model = pipeline('ner', model = model_keyword, tokenizer = tokenizer_keyword)


def get_hf_keywords(text: str, ner_model = ner_model) -> list:
    """get hugging face text keywords using hf model: bert-uncased-keyword-extractor"""    
    def sum_numbers(numbers):
      """sum numbers in a list"""
      total = 0
      for number in numbers:
        total += number
      return total

    keywords = ner_model(text)

    # add one additional fake entry at the end so we can pick up the last key phrase in the for loop below
    keywords.append({'entity': 'B-KEY', 'score': 0, 'index': 0, 'word': 'None', 'start': 0, 'end': 0})

    # Iterate keywords and create keyphrases, get average scores for appended words
    keyphrase = ''
    keyphrase_score = []
    keywords_dict = {}
    for i in range(len(keywords)):
        entity = keywords[i]["entity"]
        score = keywords[i]["score"]
        word = keywords[i]["word"]

        if entity.startswith("B"):
            # Add the previous keyphrase to final_keywords
            if keyphrase_score:
                avg_score = sum_numbers(keyphrase_score) / len(keyphrase_score)
                if "#" in keyphrase:
                    keyphrase = modify_python_str(keyphrase.strip(), removepunctuation=True, removespaces=True)
                
                # Consolidate duplicate keywords and calculate average scores
                if keyphrase in keywords_dict:
                    keywords_dict[keyphrase]["total_score"] += avg_score
                    keywords_dict[keyphrase]["count"] += 1
                else:
                    keywords_dict[keyphrase] = {"total_score": avg_score, "count": 1}

            # Reset keyphrase and keyphrase_score
            keyphrase = ''
            keyphrase_score = []

        # Append word to keyphrase and score to keyphrase_score
        keyphrase += word + " "
        keyphrase_score.append(score)

    # Convert the consolidated dictionary to a list of dictionaries with unique keywords and average scores
    final_keywords = [{"keyword": keyword, "score": keywords_dict[keyword]["total_score"] / keywords_dict[keyword]["count"]} for keyword in keywords_dict]

    return final_keywords


# spark user defined function (UDF)
keywordsUDF = udf(lambda x: get_hf_keywords(x), StringType()).asNondeterministic()

text = """Keyphrase extraction is a technique in text analysis where you extract the important keyphrases from a document.  Thanks to these keyphrases humans can understand the content of a text very quickly and easily without reading  it completely. Keyphrase extraction was first done primarily by human annotators, who read the text in detail  and then wrote down the most important keyphrases. The disadvantage is that if you work with a lot of documents,  this process can take a lot of time.  Here is where Artificial Intelligence comes in. Currently, classical machine learning methods, that use statistical  and linguistic features, are widely used for the extraction process. Now with deep learning, it is possible to capture  the semantic meaning of a text even better than these classical methods. Classical methods look at the frequency,  occurrence and order of words in the text, whereas these neural approaches can capture long-term semantic dependencies  and context of words in a text."""
print(get_hf_keywords(text))

# COMMAND ----------

# DBTITLE 1,Get Hugging Face Keywords Using a Spark Dataframe and User Defined Function

