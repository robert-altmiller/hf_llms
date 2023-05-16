# Databricks notebook source
# DBTITLE 1,Install Libraries - Transformers and Pyspark
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

# DBTITLE 1,Library Imports
import os, pathlib, re, string
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,Get Environment (Local or Databricks)
path = os.getcwd()
if "Workspace" in path:
  environment = "databricks"
else: environment = "local"
print(f"environment: {environment}")

# COMMAND ----------

# DBTITLE 1,Hugging Face Model and Token Initialization
# initial tokenizer and model
hf_model_name = "yanekyuk/bert-uncased-keyword-extractor"
print(f"initializing hf model: {hf_model_name}.....\n")
tokenizer_keyword = AutoTokenizer.from_pretrained(hf_model_name)
model_keyword = AutoModelForTokenClassification.from_pretrained(hf_model_name)
ner_model = pipeline('ner', model = model_keyword, tokenizer = tokenizer_keyword)

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

# DBTITLE 1,Get Hugging Face Keywords as a Json Object
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
    if len(final_keywords) > 0: return str(final_keywords)
    else: return str([{'keyword': "ERROR-NO-KEYWORDS", 'score': 0.0}])


# spark user defined function (UDF)
keywordsUDF = udf(lambda x: get_hf_keywords(x), StringType()).asNondeterministic()


# Keywords Function Unit Test
text = """Google is being investigated by the UK’s antitrust watchdog for its dominance in the "ad tech stack," the set of services that facilitate the sale of online advertising space between advertisers and sellers. Google has strong positions at various levels of the ad tech stack and charges fees to both publishers and advertisers. A step back: UK Competition and Markets Authority has also been investigating whether Google and Meta colluded over ads, probing into the advertising agreement between the two companies, codenamed Jedi Blue."""
print(get_hf_keywords(text))

# COMMAND ----------

# DBTITLE 1,Get Hugging Face Keywords Using a Spark Dataframe and User Defined Function
# check the environment
if environment == "databricks":

  # Create schema structures
  schema = StructType([StructField('text', StringType(), True)])

  # define some summary text data
  text1 = """
    Hugging Face is an artificial intelligence (AI) company known for its work in natural language processing (NLP). 
    The company has developed several highly influential NLP tools and models. One of Hugging Face's most significant contributions is 
    the Transformers library, an open-source library for NLP tasks. The library provides thousands of pre-trained models to perform 
    tasks on texts such as translation, summarization, text generation, classification, and more. 
    These models include BERT, GPT-2, GPT-3, T5, and many others.
    """

  text2 = """
    Google is being investigated by the UK’s antitrust watchdog for its dominance in the "ad tech stack," the set of 
    services that facilitate the sale of online advertising space between advertisers and sellers. Google has strong 
    positions at various levels of the ad tech stack and charges fees to both publishers and advertisers. A step back: 
    UK Competition and Markets Authority has also been investigating whether Google and Meta colluded over ads, 
    probing into the advertising agreement between the two companies, codenamed Jedi Blue.
    """

  text3 = """
  Hydration has different meanings depending on the context.  In the context of human biology and health, 
  hydration generally refers to the process of absorbing water. The human body needs to maintain an adequate amount of water 
  for its physiological processes such as digestion, absorption, circulation, creation of saliva, transportation of nutrients, 
  and maintenance of body temperature. In chemistry, hydration is a specific type of chemical reaction where a substance 
  combines with water. For example, when an anhydrous substance (one without water) absorbs water from its surroundings, it is said 
  to be hydrated. The attached water molecules can either be loosely attached, as in surface adsorption, or more tightly incorporated 
  into the substance's structure, forming hydrates.
  """

  # intialize spark dataframe with the text defined above
  df_data = spark.createDataFrame(data = [[text1], [text2], [text3]], schema = schema)

  # caching ensures a single udf call with spark lazy loading logic or if we need to display this dataframe
  df_data = df_data.withColumn('keywords_payload', keywordsUDF(df_data["text"])).cache()
  display(df_data)
