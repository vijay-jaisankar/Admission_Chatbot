import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import speech_recognition as sr

r = sr.Recognizer()


with open('intents.json') as file:
  data = json.load(file)

try:
	with open("data.pickle","rb") as f:
          words,labels,training,output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["intents"]:
	  for pattern in intent["patterns"]:
	    #stemming takes each word in our pattern and makes it into its root word - helpful in the training of our chatbot
	    #tokenising gets all the words in our pattern
	    wr = nltk.word_tokenize(pattern)
	    words.extend(wr)
	    docs_x.append(wr)
	    docs_y.append(intent["tag"])
	    if intent["tag"] not in labels:
	      labels.append(intent["tag"])

	words = [stemmer.stem(w1.lower()) for w1 in words if w1 != "?"]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	training = []
	output = []

	out_empty= [0 for _ in range(len(labels))]

	for x,doc in enumerate(docs_x):
	  bag = []
	  wrds = [stemmer.stem(w) for w in doc]
	  for w in words:
	    if w in wrds:
	      bag.append(1)
	    else:
	      bag.append(0)
	  output_row = out_empty[:]
	  output_row[labels.index(docs_y[x])] = 1
	  training.append(bag)
	  output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)
	with open("data.pickle","wb") as f:
          pickle.dump((words,labels,training,output),f)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net, len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
	model.load("model.tflearn")
except:
	model.fit(training, output, n_epoch = 1000, batch_size=8,show_metric = True)
	model.save("model.tflearn")


def bag_of_words(s,words):
  bag = [0 for _ in range(len(words))]
  s_words = nltk.word_tokenize(s)
  s_words = [stemmer.stem(word.lower()) for word in s_words]

  for se in s_words:
    for i,w in enumerate(words):
      if w==se:
        bag[i]=1
  return numpy.array(bag)

def chat():
  print("Start Talking to the Admissions Chatbot!(Type 'quit' to stop)")
  while True:

    print('You')
    with sr.Microphone() as source:
    print("Speak Anything")
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        inp = text
    except:
        inp = "There was an issue with recognising your voice. It's too sweet :)"
	
    if inp.lower()=="quit":
      break
    results = model.predict([bag_of_words(inp,words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    if results[results_index] > 0.65:
     for tg in data["intents"]:
       if tg["tag"] == tag:
         responses = tg['responses']
     print(random.choice(responses))
    else:
     print("I didn't get that; please try again!")

chat()
