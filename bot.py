# Smart Chatbot:
import nltk
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import string

f = open('chatbot.txt', 'r', errors='ignore')
m = open('chatbot.txt', 'r', errors='ignore')
#checkpoint = "./chatbot_weights.ckpt"
raw = f.read()
rawone = m.read()
raw = raw.lower()  # converts to lowercase
rawone = rawone.lower()  # converts to lowercase
nltk.download('punkt')  # first-time use only
nltk.download('wordnet')  # first-time use only
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words
sent_tokensone = nltk.sent_tokenize(rawone)  # converts to list of sentences
word_tokensone = nltk.word_tokenize(rawone)  # converts to list of words

sent_tokens[:2]
sent_tokensone[:2]

word_tokens[:5]
word_tokensone[:5]

lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

Introduce_Ans = ["My name is Smart Chatbot.", "My name is Smart Chatbot you can called me RISVI.", "Im Smart Chatbot :) ",
                 "My name is Smart Chatbot. and my nickname is RISVI and I am happy to solve your queries :) "]
GREETING_INPUTS = ("hello", "hi", "hiii", "hii", "hiiii", "hiiii", "greetings", "sup", "whats up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "hii there", "hi there", "hello", "I am glad! You are talking to me"]
Basic_Q = ("what is python ?", "what is python", "what is python?", "what is python.")
Basic_Ans = "Python is a high-level, interpreted, interactive and object-oriented scripting programming language python is designed to be highly readable It uses English keywords frequently where as other languages use punctuation, and it has fewer syntactical constructions than other languages."

# Checking for greetings
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Checking for Basic_Q
def basic(sentence):
    for word in Basic_Q:
        if sentence.lower() == word:
            return Basic_Ans

# Checking for Introduce
def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

#user_response =input('USER :')
#print ("BOT: My name is RISVI. Let's have a conversation! Also, if you want to exit any time, just type Bye!")
def chat(user_response):
    user_response.lower()
    keyword = " module "
    keywordone = " module"
    keywordsecond = "module "

    if (user_response != 'bye' and user_response != 'ok bye' and user_response != 'cya'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            return "You are welcome.."

        else:
            if (user_response.find(keyword) != -1 or user_response.find(keywordone) != -1 or user_response.find(
                    keywordsecond) != -1):
                return response(user_response)
                sent_tokensone.remove(user_response)
            elif (greeting(user_response) != None):
                return greeting(user_response)
            elif (user_response== "who are you" or user_response== "what is your name" or user_response=="whom am I talking to" or user_response=="tell me something about yourself"):
                return IntroduceMe(user_response)
            elif (basic(user_response) != None):
                return basic(user_response)
            else:
                return response(user_response)
                sent_tokens.remove(user_response)

    else:
        flag = False
        return "Bye! take care <3"
