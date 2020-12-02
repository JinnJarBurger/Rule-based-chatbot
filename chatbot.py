from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import bs4 as bs
import warnings
import urllib.request
import nltk
import random
import string
import re

# Uncomment these below if being run for the first time
# nltk.download('wordnet')
# nltk.download('punkt')

# to filter warnings
warnings.filterwarnings('ignore')

# getting the synonyms for the word 'hello'
synonyms = []
for syn in wordnet.synsets('hello'):
    for lem in syn.lemmas():
        lem_name = re.sub(r'\[[0-9]*\]', ' ', lem.name())
        lem_name = re.sub(r'\s+', ' ', lem.name())
        synonyms.append(lem_name)

# inputs for greeting
greeting_inputs = ['hey', 'whats up', 'good morning', 'good evening', 'morning', 'evening', 'hello there', 'hey there']
# concatenating the synonyms and the inputs for greeting
greeting_inputs = greeting_inputs + synonyms
# inputs for a normal conversation
covo_inputs = ['how are you', 'how are you doing', 'you good']
# greeting responses by the bot
greeting_responses = ['Hello! How can I help you?',
                      'Hey there! So what do you want to know?',
                      'Hi, you can ask me anything regarding Brac.',
                      'Hey! wanna know about Brac? Just ask away!']
# conversation responses by the bot
convo_responses = ['Great! what about you?', 'Getting bored at home :( wbu??', 'Not too shabby']
# conversation replies by the user
convo_replies = ['great', 'i am fine', 'fine', 'good', 'super', 'superb', 'super great', 'nice']
# few limited questions and answers given as dictionary
question_answers = {'what are you': 'I am bot, ro-bot :3',
                    'who are you': 'I am bot, ro-bot :3',
                    'what can you do': 'Answer questions regarding Brac!',
                    'what do you do': 'Answer questions regarding Brac!'}

# fetching raw html data about brac from wiki
raw_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/BRAC_(organisation)')
# processing the raw html into more readable data
raw_data = raw_data.read()

# turning html into text
article = bs.BeautifulSoup(raw_data, 'lxml')

# extracting paras from the above xml and concatenating with article_text
paragraphs = article.find_all('p')

article_text = ''

for p in paragraphs:
    article_text += p.text

article_text = article_text.lower()

# getting rid of all the special characters
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)

# extracting sentences from the article
sentences = nltk.sent_tokenize(article_text)
# extracting words from the article
words = nltk.word_tokenize(article_text)

lemma = nltk.stem.WordNetLemmatizer()


# lemmatizing words as a part of pre-processing
def perform_lemmatization(tokens):
    return [lemma.lemmatize(token) for token in tokens]


# removing punctuation
remove_punctuation = dict((ord(punc), None) for punc in string.punctuation)


# method to pre-process all the tokens utilizing the above functions
def processed_data(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(remove_punctuation)))


# function for punctuation removal
def punc_remove(str):
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ''

    for char in str:
        if char not in punctuations:
            no_punct = no_punct + char

    return no_punct


# method to generate a response to greetings
def generate_greeting_response(hello):
    if punc_remove(hello.lower()) in greeting_inputs:
        return random.choice(greeting_responses)


# method to generate a response to conversations
def generate_convo_response(str):
    if punc_remove(str.lower()) in covo_inputs:
        return random.choice(convo_responses)


# method to generate a answers to questions
def generate_answers(str):
    if punc_remove(str.lower()) in question_answers:
        return question_answers[punc_remove(str.lower())]


# method to generate response to queries regarding brac
def generate_response(user):
    bracrobo_response = ''
    sentences.append(user)

    word_vectorizer = TfidfVectorizer(tokenizer=processed_data, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched is 0:
        bracrobo_response = bracrobo_response + 'Sorry, my database doesn\'t have the response for that. Try ' \
                                                'something different and related to Brac. '
        return bracrobo_response
    else:
        bracrobo_response = bracrobo_response + sentences[similar_sentence_number]
        return bracrobo_response


# chatting with the chatbot -->
continue_chat = True
print('Hi! I am BracRobo. You can ask me anything regarding Brac and I shall try my best to answer them: ')
while continue_chat:
    user_input = input().lower()
    user_input = punc_remove(user_input)
    if user_input != 'bye':
        if user_input == 'thanks' or user_input == 'thank you very much' or user_input == 'thank you':
            continue_chat = False
            print('BracRobo: Not a problem! (And WELCOME! :D)')
        elif user_input in convo_replies:
            print('That\'s nice! How may I be of assistance?')
            continue
        else:
            if generate_greeting_response(user_input) is not None:
                print('BracRobo: ' + generate_greeting_response(user_input))
            elif generate_convo_response(user_input) is not None:
                print('BracRobo: ' + generate_convo_response(user_input))
            elif generate_answers(user_input) is not None:
                print('BracRobo: ' + generate_answers(user_input))
            else:
                print('BracRobo: ', end='')
                print(generate_response(user_input))
                sentences.remove(user_input)
    else:
        continue_chat = False
        print('BracuRobo: Bye, take care, stay home and stay safe!')
