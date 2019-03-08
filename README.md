# AI-project  

## DESCRIPTION
This is a chatbot named Yue that can answer questions on stock price, trading volume and marketing value. 

The chatbot is integrated into Wechat, using iexfiance API.

Users are only allowed to send text or audio to ask questions.    

## ACHIEVED GOALS  
Multiple selective answers to the same question and provides a default answer.  
Chatbot can answer questions through regular expressions, pattern matching, keyword extraction, syntax conversion.  
It can extract users' intents through regular expressions.  
The user’s entities can be extracted by a support vector machine. Based on predefined entities’ types to identify an entity.   
Construct a local basic chat robot system through Rasa NLU to explore the database using natural language.  
Implement multiple rounds of multi-query technology for state machines and provide explanations and answers based on contextual issues.  The chatbot can achieve receiving the audio from the user and reply the information using Google Speech Recognition.


## DEMO
Text test and audio test show the results of this product.  

Below is the demo of text test. The user ask question through text.  

![](text.gif)   

https://youtu.be/VchJBKMimYo  

This demo is the audio test from chatbot perspective. The user ask audio questions.   

![](audio.gif)  

https://youtu.be/5-321E-vq4k

## TECHNOLOGY USED
https://pypi.org/project/iexfinance/  

https://docs.python.org/3/library/re.html  

https://spacy.io/  

https://rasa.com/docs/nlu/  

https://github.com/jiaaro/pydub  

https://github.com/youfou/wxpy  









