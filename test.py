#dict = { 
    #'name': 'andra',
    #'country': "uk"   
#}
# import pprint 
# pprint.pprint(dict)
# dict.update({'name': "maria"})
# pprint.pprint(dict)

#print(dict.get("namew", None))
#dict
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

#nltk.download('punkt_tab')
import nltk
from nltk.tokenize import word_tokenize

text = "The Prime Minister addressed the nation today."
print(word_tokenize(text))