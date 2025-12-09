import os
import numpy as np
import pandas as pd

from os.path import join
from transformers import AutoTokenizer, TFAutoModel

def load_data(file_list, motion_data_dir, text_data_dir):
    data = {}
    with open(file_list, 'r') as f:
        titles = f.read().splitlines()
        for t in titles[:10]:
            npy_file = f"{t}.npy"
            motion_data = np.load(join(motion_data_dir, npy_file))
            
            # Charger et traiter la description
            txt_file = f"{text_data_dir}{t}.txt"
            with open(txt_file, 'r', encoding='utf-8') as m:
                desc = m.readline().split('#')[0].capitalize()
            
            data[desc] = motion_data
    
    return pd.DataFrame({'Description': list(data.keys()), 'Motion': list(data.values())})

def pad_motion_sequences(motions, max_length):
    T = len(motions)
    N = motions[0].shape[1]  # N
    d = motions[0].shape[2]  # d

    # Initialisation du tenseur avec des zéros
    mtpadded = np.zeros((T, max_length, N, d))

    for i, motion in enumerate(motions):
        T = motion.shape[0]  # Longueur réelle de la séquence
        mtpadded[i, :T, :, :] = motion  # Copier la séquence dans le tenseur

    return mtpadded

def encode_text(texts):
    return tknz(texts, padding=True, truncation=True, max_length=50, return_tensors='tf')

# Répertoires des données
motion_data_dir = "./motions/"
text_data_dir = "./texts/"

# Charger les ensembles de données
traindf = load_data('train.txt', motion_data_dir, text_data_dir)
valdf = load_data('val.txt', motion_data_dir, text_data_dir)

xtrain, ytrain = traindf['Motion'], traindf['Description']
xval, yval = valdf['Motion'], valdf['Description']

# Configuration des paramètres
hidden_units = 256
max_motion = max([x.shape[0] for x in xtrain])

# Initialisation du Tokenizer BERT
tknz = AutoTokenizer.from_pretrained('bert-base-uncased')

# Transformer les descriptions en vecteurs BERT
ytrainenc = encode_text(ytrain.tolist())
yvalenc = encode_text(yval.tolist())

# Padding des séquences de mouvement
xtrainpad = pad_motion_sequences(xtrain, max_motion)
xvalpad = pad_motion_sequences(xval, max_motion)

xtrainpad = np.reshape(xtrainpad, (xtrainpad.shape[0], xtrainpad.shape[1], -1))  
xvalpad = np.reshape(xvalpad, (xvalpad.shape[0], xvalpad.shape[1], -1)) 
print("Forme des données de mouvement (train) :", xtrainpad.shape)
print("Forme des données de mouvement (validation) :", xvalpad.shape)

# Encoder LSTM pour les poses 3D
motion_input = Input(shape=(max_motion, 22, 3))  # (nb_frames, features)

encoder_lstm = LSTM(hidden_units, return_state=True)
_, state_h, state_c = encoder_lstm(motion_input)
encoder_states = [state_h, state_c]

# Créer un modèle séparé pour l'encodeur
encoder_model = Model(motion_input, encoder_states)

# Charger le modèle BERT pour l'encodage du texte
bert = TFAutoModel.from_pretrained('bert-base-uncased')

# Entrée du texte encodé par BERT
text_input = Input(shape=(None,), dtype=tf.int32)
bert_output = bert(text_input)[0]  # La sortie de BERT
text_lstm = LSTM(hidden_units, return_sequences=True)(bert_output)

decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_state_input_h = Input(shape=(hidden_units,))
decoder_state_input_c = Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_output, dec_h, dec_c = decoder_lstm(text_lstm, initial_state=decoder_states_inputs)
dec_states = [dec_h, dec_c]

decoder_dense = Dense(tknz.vocab_size, activation='softmax')
output = decoder_dense(decoder_output)

decoder_model = Model([text_input] + decoder_states_inputs, [output] + dec_states)

# Définir le modèle final
model = Model([motion_input, text_input], output)
model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit([xtrainpad, ytrainenc['input_ids']], ytrainenc['input_ids'], batch_size=32, epochs=20)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

bleu_scores = []
smoothie = SmoothingFunction().method1  # Pour éviter BLEU=0 sur de petites phrases

for i, motion in enumerate(xval):
    if i>=5:
        break
    input_seq = pad_motion_sequences(np.expand_dims(motion, axis=0), max_motion)
    input_seq = tf.convert_to_tensor(input_seq, dtype=tf.float32)
    state_h, state_c = encoder_model.predict(input_seq)  # Correct init encodeur

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tknz.convert_tokens_to_ids(['[CLS]'])[0]  # Token de départ

    decoded_sentence = []
    for _ in range(50):  # Maximum 50 tokens
        output_tokens, state_h, state_c = decoder_model.predict([target_seq] + [state_h, state_c])
        output_word = np.argmax(output_tokens[0, -1, :])
        word = tknz.convert_ids_to_tokens([output_word])[0]

        if word == '[SEP]' or word == '':
            break

        decoded_sentence.append(word)
        target_seq[0, 0] = output_word  # Ajouter le mot généré

    print(f"Phrase prédite {i+1}: {' '.join(decoded_sentence)}")