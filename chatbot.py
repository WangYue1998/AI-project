from wxpy import *
import random
import re


rules = {'I want (.*)': ['What would it mean if you got {0}', 
                         'Why do you want {0}', 
                         "What's stopping you from getting {0}"], 
         'Do you remember (.*)': ['Did you think I would forget {0}', 
                                  "Why haven't you been able to forget {0}", 
                                  'What about {0}', 
                                  'Yes .. and?'], 
         'Do you think (.*)': ['if {0}? Absolutely.', 
                               'No chance'], 
         'if (.*)': ["Do you really think it's likely that {0}", 
                     'Do you wish that {0}', 
                     'What do you think about {0}', 
                     'Really--if {0}']
        }

# Define match_rule()
def match_rule(rules, message):
    response, phrase = "I don't know :(", None
    
    # Iterate over the rules dictionary
    for key,value in rules.items():
        # Create a match object
        match =re.search(key,message)
        if match is not None:
            # Choose a random response
            response = random.choice(rules[key])
            if '{0}' in response:
                phrase = match.group(1)
    # Return the response and phrase
    return response, phrase


# Define replace_pronouns()
def replace_pronouns(message):

    message = message.lower()
    if 'me' in message:
        # Replace 'me' with 'you'
        return re.sub('me','you',message)
    if 'my' in message:
        # Replace 'my' with 'your'
        return re.sub('my','your',message)
    if 'your' in message:
        # Replace 'your' with 'my'
         return re.sub('your','my',message)
    if 'you' in message:
        # Replace 'you' with 'me'
        return re.sub('you','me',message)

    return message

# Define respond()
def respond(message):
    # Call the match_intent function  "greeting"
    intent = match_intent(message)
    if intent is not None:
        return responses[intent]
    # Call match_rule
    response, phrase = match_rule(rules, message)
    if '{0}' in response:
        # Replace the pronouns in the phrase
        phrase = replace_pronouns(phrase)
        # Include the phrase in the response
        response = response.format(phrase)
        return response
        
    
keywords = {
            'greet': ['hello', 'Hi', 'hi','Hello','hey',"what's up"], 
            'thankyou': ['thank you', 'Thank you','thx','Thanks'], 
            'goodbye': ['bye', 'farewell','88']
           }
# Define a dictionary of patterns
patterns = {}
# Iterate over the keywords dictionary
for intent, keys in keywords.items():
    # Create regular expressions and compile them into pattern objects
    patterns[intent] = re.compile("|".join(keys))


responses = {'greet': 'Hello you! :)', 
             'thankyou': 'you are very welcome', 
             'default': 'default message', 
             'goodbye': 'goodbye for now'
            }   
# Define a function to find the intent of a message
def match_intent(message):
    matched_intent = None
    for intent, pattern in patterns.items():
        # Check if the pattern occurs in the message 
        if pattern.search(message):
            matched_intent = intent
    return matched_intent

import spacy
# Import SVC
from sklearn.svm import SVC
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
def entity_train():
    # Load the spacy model: nlp, en_core_web_md
    nlp = spacy.load("en_core_web_md")
    # Create a support vector classifier
    clf = SVC()
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')['label']
    # Fit the classifier using the training data
    clf.fit(X_train, y_train)
    return nlp

from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
'''training for interpreter'''
def interpreter_train():
    # Create a trainer
    trainer = Trainer(config.load("config_spacy.yml"))
    # Load the training data
    training_data = load_data('wy.json')
    # Create an interpreter by training the model
    interpreter = trainer.train(training_data)
    return interpreter

def interprete(interpreter,message):
    intp=interpreter.parse(message)
    return intp


# Define extract_entities()
def extract_entities(nlp, message):
    # Define included entities
    include_entities = ['DATE', 'ORG', 'PERSON']
    # Create a dict to hold the entities
    ents = dict.fromkeys(include_entities)
    # Create a spacy document
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ in include_entities:
            # Save interesting entities
            ents[ent.label_] = ent.text
    return ents

def check_Org(entities):
    if entities['ORG'] is not None:
        return entities['ORG']
    return None

'''assign each intent to a letter and return the letter of user's intent '''
def check_intents(intent):
    intents=[['a','greet'],['b','price search'],[
             'c','trading volume search'],['d','market value search'],
             ['e','appreciate'],['f',None],['g','quit'],['h','specify company'],
             ['i','confirm'],['j','deny']]
    for i in intents:
        if i[1]==intent:
            return i[0]  
        
'''pending actions according to pending type'''
def pending_actions():
    dic={0:'Which company are you asking? Please say "The name is...".',
     1:'What would you like to ask?',     
     2:'Sorry, I dont understand. Maybe you have typed wrong letter or words.' ,
     3:'What do you want to ask about this company?'
     }
    return dic

'''actions at each state'''
def state_change_action(state):
    dic={ 0:'',   
     1:'Which company are you asking? Please say "The name is..."',
     2:"Company specified",
     3:"Ok, I have found it." ,
     4:"Sorry, I cannot find that information"   
     }
    answer=dic[state]
    return answer

'''if there is an error when searching skip the error.'''
def error_check1(CPN):
    try:
        p_CPN=CPN.get_price()
    except Exception:
        error='information of company not found'
        return error
    else:
        return 1
def error_check2(CPN):
    try:
        v_CPN=CPN.get_volume()
    except Exception:
        error='information of company not found'
        return error
    else:
        return 1        
def error_check3(CPN):
    try:
        c_CPN=CPN.get_market_cap()
    except Exception:
        error='information of company not found'
        return error
    else:
        return 1  
  
    
from iexfinance import Stock   
import requests
def get_ticker_symbol(name):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(name)
    result = requests.get(url).json()
    for x in result['ResultSet']['Result']:
        if x is not None:
            return x['symbol']
        else:
            return 0


def bot_reply(intent,state,pending,Org,pending_action,d): #d dictionary
    if state==0:#INIT
        if intent in 'bcd' and Org is not None:
            pending,pending_action,state=0,None,2
            d['company'],d['function']=Org,intent
        elif intent in 'bcd' and Org is None:
            pending,pending_action,state=0,None,1
            d['company'],d['function']=Org,intent     
        elif intent not in 'bcd' and Org is not None:
            d['company'],d['function']=Org,None
            pending,pending_action,state=1,3,2
        else:
            pending,pending_action,state=1,1,0
            d['company'],d['function']=None,None
    elif state==1:#Specify company
        if (intent=='h') and (d['company'] is None) and (Org is not None):
            pending,pending_action,state=0,None,2
            d['company']=Org
          
        elif intent=='h' and d['company'] is not None :
            pending,pending_action,state=0,None,2          
            
        elif intent in 'bcd'and Org is not None:
            pending,pending_action,state=0,None,2
            d['company']=Org
            d['function']=intent
        elif intent=='e'or'f'or'i':
            pending,pending_action,state=1,2,1       
    
    elif state==2:#company specified      
        if intent in 'bcd':
            if Org is not None:#
                pending,pending_action,state=0,None,1
                d['company']=Org
                d['function']=intent            
            elif d['company'] is not None and d['function'] is None and Org is None:#Company specified but question not specified
                pending,pending_action,state=0,None,2
                d['function']=intent    

            
    elif state==3:#information found
        if intent=='j':  # no 
            state=5
        elif intent=='i':  # thanks
            pending,pending_action,state=1,1,0
            d['company']=None
            d['function']=None
        elif intent in 'bcd'and Org is None:
            if d['company'] is not None:
                 pending,pending_action,state=0,None,2
                 d['function'] = intent
            else:
                pending,pending_action,state=0,None,1
                d['company'],d['function']=Org,intent
        elif intent in 'bcd' and Org is not None:
            pending,pending_action,state=0,None,2
            d['company'],d['function']=Org,intent
        else:
            pending,pending_action,state=1,1,0
            d['company'],d['function']=None,None
    
    return d,pending,pending_action,state 

from pydub import AudioSegment
import speech_recognition as sr
r = sr.Recognizer()
from io import BytesIO
def txt_recog(msg):

        audio = AudioSegment.from_file(BytesIO(msg.get_file()))
        export = audio.export('file.wav', format="wav")#change the path
        AUDIO_FILE = 'file.wav'#The same path as above
        user='empty'
        with sr.AudioFile(AUDIO_FILE) as source:
            print('say something')
            audio = r.record(source)  # read the entire audio file
        try:
            user=r.recognize_google(audio)
            print("Google Speech Recognition thinks you said " + user)  
            return user
        
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        
        


      
if __name__ == "__main__": 
        bot = Bot()
        my_friend = bot.friends().search('Yue')[0]   
        
        nlp=entity_train() 
        interpreter=interpreter_train()  
       
        dictionary={'company':None,'function':None}
        pending=0
        pending_action=None
        #numerical representation of state
        INIT,SPECIFY_COMPANY,SPECIFIED,FOUND,NOTFOUND,END=0,1,2,3,4,5
        state=INIT
        
        my_friend.send("Welcome to Yue's stock chatbot! I can help you to check the price, trading volume, and market value of companies.")
        
        @bot.register(my_friend)
        def reply_my_friend(msg):
            global dictionary,pending,pending_action,state,my_friend
            if msg.type =='Recording':
                user=txt_recog(msg)
                if user is None:
                    my_friend.send(('Google Speech Recognition could not understand audio'))
                    return
            else:
                user = msg.text
            
            if user== "quit" or state ==END:
                my_friend.send(('See you.'))
                return
            if respond(user) != None:
                my_friend.send(respond(user))
                return
            
            intent=interprete(interpreter,user)['intent']['name']
#            print(intent)
            entities=extract_entities(nlp,user)
            Org=check_Org(entities)
#            print(Org)
            intent_=check_intents(intent)     
#           print(intent_)
            dictionary,pending,pending_action,state=bot_reply(intent_,state,pending,Org,pending_action,dictionary)
#            print(pending)
#            print(pending_action)
#            print(state)
            action=pending_actions()       

            if pending is not 0:
                my_friend.send(action[pending_action]) 
                return
            if state is not 0 and state is not 3 and state is not 5 and state is not 2:
                my_friend.send(state_change_action(state))
            if (dictionary['function'] is not None) and (dictionary['company']is not None):
                CPN=Stock(get_ticker_symbol(dictionary['company']))
                if dictionary['function']=='b':               
                    if error_check1(CPN)==1:      
                        p_CPN=CPN.get_price()
                        state=FOUND
                        my_friend.send('Ok,I have found it! The price of {0} is ${1}'.format(dictionary['company'],p_CPN) )
                    else:     
                        my_friend.send('Price Information of company not found')
                        state=NOTFOUND
                if dictionary['function']=='c':
                    if error_check2(CPN)==1:                
                        mv_CPN=CPN.get_volume()
                        my_friend.send('Ok,I have found it!')
                        state=FOUND
                        my_friend.send('The volume of {0} is {1}'.format(dictionary['company'],mv_CPN) )
                    else: 
                        
                        my_friend.send('Volume Information of company not found')
                        state=NOTFOUND            
                if dictionary['function']=='d':
                    if error_check3(CPN)==1:                
                        mc_CPN=CPN.get_market_cap()
                        my_friend.send('Ok,I have found it!')
                        state=FOUND
                        my_friend.send('The market value of {0} is ${1}'.format(dictionary['company'],mc_CPN))  
                    else:                    
                        my_friend.send('Market Value Information of company not found')
                        state=NOTFOUND
            if state==NOTFOUND :
                my_friend.send('Do you have other questions? Say "quit" to quit chat!')
                state=INIT
            if state == FOUND:
                my_friend.send('Do you have other questions? Say "quit" to quit chat!')
#            print(state)

            




    