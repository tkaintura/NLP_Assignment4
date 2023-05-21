from spacy import displacy
import spacy_stanza
import stanza
import spacy

# # Download the English and Greek model for stanza, these are the versions 2 treebanks
# stanza.download('el', package='gdt')
# stanza.download('en', package='gum')

# Build Pipeline and run the Greek model
nlp_gr = stanza.Pipeline('el', package='gdt')
doc_gr = nlp_gr('Αυτή είναι μία τυχαία πρόταση.')
displacy.render(doc_gr, style="dep")

# Build Pipeline and run the English model
nlp_en = stanza.Pipeline('en', package='gum')
doc_en = nlp_en('The university blocked the acquisition, citing concerns about the risks involved.')
displacy.serve(doc_en)


#THESE TWO PIPELINES ABOVE WON'T WORK!

# This works but again it's going to show you the v2 version of the treebank.
# Open 127.0.0.1:5000 in your browser to see the dependency tree!
nlp = spacy_stanza.load_pipeline('en', package='gum', processors = 'tokenize,mwt,pos,lemma,depparse')
doc = nlp("The university blocked the acquisition, citing concerns about the risks involved.")
displacy.serve(doc)



###
# StandfordNLP library (with the old treebanks)
###

import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
# stanfordnlp.download('en', version='0.1.0')

snlp = stanfordnlp.Pipeline(lang="en", models_dir="C:/Users/antge/stanfordnlp_resources")
nlp = StanfordNLPLanguage(snlp)
doc = nlp("The university blocked the acquisition, citing concerns about the risks involved.")
displacy.serve(doc)







# from stanfordnlp.pipeline import Stanford

# # Initialize the NLP pipeline
# nlp = StanfordCoreNLPPipeline()

# # Define the sentence
# sentence = "The quick brown fox jumps over the lazy dog."

# # Parse the sentence 
# doc = nlp(sentence)

# # Get the dependency parse 
# parse = doc.sentences[0].dependencies 

# # Render the parse using Universal Dependencies notation
# text = parse.render(style="universal")

# # Print the parse 
# print(text)

# import os

# # Specify the path to your CoreNLP Java checkout
# corenlp_home = 'C:\\Users\\antge\\Desktop\\CoreNLP'

# # Set the environment variable
# os.environ['CORENLP_HOME'] = corenlp_home


# from stanfordnlp.server import CoreNLPClient

# #Specify the path to the model or treebank directory
# model_path = 'C:\\Users\\antge\\stanfordnlp_resources\\en_ewt_models'

# # Create a CoreNLPClient object and pass the model path
# with CoreNLPClient(annotators='tokenize,pos', models_path=model_path) as client:
#     # Use the client object to process your text
#     text = "Your text goes here."
#     ann = client.annotate(text)

