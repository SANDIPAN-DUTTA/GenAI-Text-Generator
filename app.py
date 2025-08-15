import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

text = """
First and foremost, Lionel Messi is an incredible footballer.
His natural talent and ability to read the game are second to none, and he has consistently shown himself to be a world-class performer.
On the pitch, he is a magician with the ball at his feet, capable of scoring goals and creating chances for his teammates with equal ease.
Lionel Messi, one of the greatest soccer players of all time, has announced that he will be retiring from international soccer after the 2022 World Cup.
This comes as a surprise to many fans and analysts, who expected Messi to continue playing for Argentina for years to come.
In this blog post, we’ll take a look at Messi’s career and why he might be making this decision.
We’ll also speculate on who could replace him as the face of Argentine soccer.
My favorite sport is football. The club that I support is – FC Barcelona.
Lionel Messi is my favorite footballer. He is also my favorite sportsperson.
This essay is dedicated to Lionel Messi.
Lionel Messi is a legend of the modern game.
He is considered to be one of the greatest players of all time.
Alongside Cristiano Ronaldo, Lionel Messi is widely regarded as one of the two best players of all time.
But by far Messi is still the Favorite Sportsperson of all time!!
Coaches, pundits, and teammates describe him as a ‘magician’ and ‘alien’.
Yes, he managed to perform extraordinary things on the field. I’ve been playing football since my childhood.
When I was a kid, my parents used to take me to the nearest park, where I’d kick a football around.
Since then, I’ve been in love with this game. Gradually, I started playing football with my friends and other kids in the neighborhood.
Such was our love for the game that we even formed a local club. We would take part in neighborhood and local football matches and tournaments.
What I’m trying to say is that I’ve been a huge fan of football since my childhood.
The game has played an important role in shaping up my childhood and character.
For example, football has taught me the importance of health and physical conditioning. It has also taught me the importance of sportsmanship and respect.
 """
# clean the text data
text = text.lower().replace('\n', ' ')

# Tokenizing the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_word = len(tokenizer.word_index) + 1

# Creting input sequence
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Creating predictor and labels
predictors , label = input_sequences[:, :-1], input_sequences[:, -1]
label = to_categorical(label, num_classes=total_word)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , Bidirectional , LSTM , Dense , Dropout

model = Sequential()
model.add(Embedding(total_word, 100, input_length=max_sequence_len-1))
model.add(LSTM(150 , return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_word , activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)
        predict_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# example usage
print(generate_text("Lionel Messi", 5, model, max_sequence_len))

from transformers import GPT2LMHeadModel , GPT2Tokenizer

model_name = 'gpt2'  # you can use 'gpt2-medium' , gpt2-large , etc. for larger models
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_text(seed_text, max_length=100):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1,
                            no_repeat_ngram_size = 2 , early_stopping=True)
    generated_text = tokenizer.decode(output[0] , skip_special_tokens=True)
    return generated_text

seed_text = "Sandipan Dutta"
generated_text = generate_text(seed_text , max_length=100)
print(generated_text)