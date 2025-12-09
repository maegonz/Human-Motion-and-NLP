import os
import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


# Charger les ensembles de données
def load_data(file_list, motion_data_dir, text_data_dir):
    data = {}
    with open(file_list, 'r') as f:
        titles = f.read().splitlines()
        for t in titles:
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

# Répertoires des données
motion_data_dir = "./motions/"
text_data_dir = "./texts/"

# Charger les ensembles de données
traindf = load_data('train.txt', motion_data_dir, text_data_dir)
valdf = load_data('val.txt', motion_data_dir, text_data_dir)

# Affichage des tailles des ensembles
print('Train:', traindf.shape)
print('Validation:', valdf.shape)

xtrain, ytrain = traindf['Motion'], traindf['Description']
xval, yval = valdf['Motion'], valdf['Description']

# Configuration des paramètres
hidden_units = 256
emb_dim = 128
max_motion = max([x.shape[0] for x in xtrain] )

# Tokenizer
tknz = Tokenizer()
tknz.fit_on_texts(ytrain.tolist() + ['startseq', 'endseq'])
vocab_size = len(tknz.word_index) + 1

# Transformer les descriptions en vecteurs
ytrainseq = tknz.texts_to_sequences(ytrain)
yvalseq = tknz.texts_to_sequences(yval)
max_seq = max(len(seq) for seq in ytrainseq)

# Padding
xtrainpad = pad_motion_sequences(xtrain, max_motion)
xvalpad = pad_motion_sequences(xval, max_motion)

xtrainpad = np.reshape(xtrainpad, (xtrainpad.shape[0], xtrainpad.shape[1], -1))  # Forme (x, x, 66)
xvalpad = np.reshape(xvalpad, (xvalpad.shape[0], xvalpad.shape[1], -1))
print("Forme des données de mouvement (train) :", xtrainpad.shape)
print("Forme des données de mouvement (validation) :", xvalpad.shape)

ytrainpad = pad_sequences(ytrainseq, maxlen=max_seq)
yvalpad = pad_sequences(yvalseq, maxlen=max_seq)
print("Forme des données de texte (train) :", ytrainpad.shape)
print("Forme des données de texte (validation) :", yvalpad.shape)

# Encoder LSTM pour les poses 3D
motion_input = Input(shape=(max_motion, 22, 3))  # (nb_frames, features)
reshape = tf.keras.layers.Reshape((max_motion, 66))(motion_input)

encoder_lstm = LSTM(hidden_units, return_state=True)
_, state_h, state_c = encoder_lstm(reshape)
encoder_states = [state_h, state_c]

encoder_model = Model(motion_input, encoder_states)

# Décodeur pour le texte
text_input = Input(shape=(max_seq,))
embedding = Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True)(text_input)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder_lstm(embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
output = decoder_dense(decoder_output)

# Modèle final
model = Model([reshape, text_input], output)
model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('OK')

# Entrées du modèle de décodage
decoder_input = Input(shape=(1,))  # Un seul token à la fois
decoder_state_input_h = Input(shape=(hidden_units,))
decoder_state_input_c = Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Embedding du token actuel
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True)(decoder_input)

# LSTM du décodeur (utilisation en inférence)
decoder_output, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

# Couche de sortie
decoder_output = decoder_dense(decoder_output)

# Modèle du décodeur pour la prédiction
decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_output] + decoder_states)

print('OK')

# Entraînement
model.fit([xtrainpad, ytrainpad], ytrainpad,
          batch_size=32, epochs=20)

def temp_sample(output_tokens, temperature=1.0):
    # Applique une température sur les logits pour obtenir des probabilités plus "lisses"
    output_tokens = np.asarray(output_tokens).astype('float64')
    output_tokens = np.log(output_tokens + 1e-10) / temperature
    proba = np.exp(output_tokens) / np.sum(np.exp(output_tokens))
    
    # Utilisation de la méthode de sampling
    sampled_token_index = np.random.choice(len(proba[0, -1, :]), p=proba[0, -1, :])
    return sampled_token_index


for i in range(2):
    input_seq = pad_motion_sequences(np.expand_dims(xval[i], axis=0), max_motion)
    # Encoder : Extraire les états cachés
    input_seq = tf.convert_to_tensor(input_seq, dtype=tf.float32)
    states_value = encoder_model.predict(input_seq)  

    # Début de la séquence (token de départ)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tknz.word_index['startseq']

    # Stockage de la phrase générée
    decoded_sentence = []

    for _ in range(max_seq-10):
        # Prédiction d'un mot
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Choisir le mot avec la plus grande probabilité
        # sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token_index = temp_sample(output_tokens, temperature=0.8)
        sampled_word = tknz.index_word.get(sampled_token_index, '')

        # Arrêter si "endseq" est généré
        if sampled_word == "endseq":
            break

        decoded_sentence.append(sampled_word)

        # Mise à jour du token d'entrée
        # target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Mise à jour des états cachés du décodeur
        states_value = [h, c]

    print(f"Phrase prédite {i+1}: {' '.join(decoded_sentence)}")
