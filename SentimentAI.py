# import pandas as pd
# import numpy as np
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from gensim.models import Word2Vec
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import os
# import pickle


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

# MAX_SEQUENCE_LENGTH = 35
# EMBEDDING_DIM = 70 # From Word2Vec vector_size
# NUM_HEADS = 3
# FF_DIM = 120
# NUM_CLASSES = 3 # For negative, neutral, positive (-1, 0, 1)

# class PositionalEmbedding(layers.Layer):
#     def __init__(self, sequence_length, vocab_size, embed_dim, token_initial_weights=None, **kwargs):
#         super().__init__(**kwargs)
#         if token_initial_weights is not None:
#             # Initialize with pre-trained weights, and optionally make it non-trainable
#             self.token_embeddings = layers.Embedding(
#                 input_dim=vocab_size,
#                 output_dim=embed_dim,
#                 mask_zero=True,
#                 weights=[token_initial_weights],
#                 trainable=False # Typically, pre-trained embeddings are not trained further, or at least not initially
#             )
#         else:
#             self.token_embeddings = layers.Embedding(
#                 input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
#             )
#         self.position_embeddings = layers.Embedding(
#             input_dim=sequence_length, output_dim=embed_dim
#         )
#         self.sequence_length = sequence_length
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim

#     def call(self, inputs):
#         length = tf.shape(inputs)[-1]
#         positions = tf.range(start=0, limit=length, delta=1)
#         embedded_tokens = self.token_embeddings(inputs)
#         embedded_positions = self.position_embeddings(positions)
#         return embedded_tokens + embedded_positions
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "sequence_length": self.sequence_length,
#             "vocab_size": self.vocab_size,
#             "embed_dim": self.embed_dim,
#         })
#         return config
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    
#     def build(self, input_shape):
#         super().build(input_shape)

    



# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
#         self.ffn = keras.Sequential(
#             [
#                 layers.Dense(ff_dim, activation="relu"),
#                 layers.Dense(embed_dim),
#             ]
#         )
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         # self.dropout2 = layers.Dropout(rate)
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.ff_dim = ff_dim
#         self.rate = rate

#     def call(self, inputs, training=None):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         # ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "embed_dim": self.embed_dim,
#             "num_heads": self.num_heads,
#             "ff_dim": self.ff_dim,
#             "rate": self.rate,
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    
#     def build(self, input_shape):
#         super().build(input_shape)



# class SentimentAI:
#     def __init__(self):
#         self.model = self.load_model()  # Attempt to load pre-trained model if available
#         preprocessing_loaded = self.load_preprocessing()
#         if self.model is None or not preprocessing_loaded:
#             train_df = pd.read_csv("train.csv", encoding="latin1")
#             test_df = pd.read_csv("test.csv", encoding="latin1")

#             self.stop_words = set(stopwords.words('english'))

#             # Combine text for Word2Vec training
#             all_raw_text = pd.concat([train_df['text'], test_df['text']], axis=0).astype(str).tolist()
#             self.word2vec_model, self.word_index, self.embedding_matrix = self._train_word2vec_model(all_raw_text)
#             self.save_preprocessing()

#             self.train_data = self.preprocess_data(train_df)
#             self.test_data = self.preprocess_data(test_df)

#             self.model = self.train_model(self.train_data)
#             self.save_model()  # Save the trained model for future use

#     def _clean_text(self, text):
#         tokens = word_tokenize(str(text))
#         tokens = [word.lower() for word in tokens if word.isalpha()]
#         # tokens = [word for word in tokens if word not in self.stop_words]
#         tokens = [word for word in tokens if len(word)>1]
#         return tokens

#     def _train_word2vec_model(self, all_raw_text):
#         # Tokenize all text for Word2Vec
#         tokenized_sentences = [self._clean_text(text) for text in all_raw_text]

#         # Train Word2Vec model
#         word2vec_model = Word2Vec(
#             sentences=tokenized_sentences,
#             vector_size=EMBEDDING_DIM,
#             window=5,
#             min_count=2,
#             workers=4
#         )

#         # Create word_index and embedding_matrix
#         word_index = {"PAD": 0} # 0 for padding
#         for i, word in enumerate(word2vec_model.wv.index_to_key):
#             word_index[word] = i + 1  # Start from 1

#         vocab_size = len(word_index)
#         embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
#         for word, i in word_index.items():
#             if word != "PAD": # Skip PAD, it's already 0
#                 embedding_matrix[i] = word2vec_model.wv[word]

#         return word2vec_model, word_index, embedding_matrix


#     def preprocess_data(self, data):
#         data = data.drop(columns=[
#             'textID', 'selected_text', 'Time of Tweet',
#             'Age of User', 'Country', 'Population -2020',
#             'Land Area (Km²)', 'Density (P/Km²)'
#         ], errors='ignore')

#         data["sentiment"] = data["sentiment"].map({
#             'positive': 2, # Map to 0, 1, 2 for SparseCategoricalCrossentropy
#             'neutral': 1,
#             'negative': 0
#         })

#         data = data.dropna(subset=['text', 'sentiment']).reset_index(drop=True)

#         # Apply cleaning to 'text' column, storing tokenized lists
#         data["text"] = data["text"].apply(self._clean_text)

#         return data

#     def _texts_to_sequences(self, texts):
#         sequences = []
#         for text_tokens in texts:
#             sequence = [self.word_index.get(word, 0) for word in text_tokens]
#             sequences.append(sequence)
#         return keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')


#     def train_model(self, data):
#         X = self._texts_to_sequences(data["text"])
#         y = data["sentiment"].values

#         inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH,))

#         # Use Embedding layer initialized with Word2Vec weights passed during initialization
#         embedding_layer = PositionalEmbedding(
#             MAX_SEQUENCE_LENGTH,
#             len(self.word_index),
#             EMBEDDING_DIM,
#             token_initial_weights=self.embedding_matrix
#         )
#         x = embedding_layer(inputs)

#         # Transformer Blocks
#         for _ in range(2): # Use 2 transformer blocks
#             x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, FF_DIM)(x)

#         x = layers.GlobalAveragePooling1D()(x) # Corrected typo
#         # x = layers.Dropout(0.1)(x)
#         x = layers.Dense(10, activation="relu")(x)
#         # x = layers.Dropout(0.1)(x)
#         outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

#         model = keras.Model(inputs=inputs, outputs=outputs)

#         model.compile(
#             optimizer="adam",
#             loss="sparse_categorical_crossentropy",
#             metrics=["accuracy"]
#         )

#         print("Training Transformer model...")
#         model.fit(
#             X,
#             y,
#             batch_size=64,
#             epochs=1, # Reduced epochs for faster execution in example
#             validation_split=0.1
#         )
#         print("Transformer model trained.")
#         return model
    
#     def save_model(self, model_path="sentiment_transformer_model.keras"):
#         self.model.save(model_path,include_optimizer=False)
    
#     def load_model(self, model_path="sentiment_transformer_model.keras"):
#         model_path = os.path.abspath(model_path)

#         if not os.path.exists(model_path):
#             print("No pre-trained model found. A new model will be trained.")
#             return None

#         try:
#             return keras.models.load_model(
#                 model_path,
#                 custom_objects={
#                     "PositionalEmbedding": PositionalEmbedding,
#                     "TransformerBlock": TransformerBlock
#                 }
#             )
#         except Exception as e:
#             print("Error loading model:", e)
#             return None
        
#     def save_preprocessing(self):
#         with open("word_index.pkl", "wb") as f:
#             pickle.dump(self.word_index, f)

#         with open("embedding_matrix.pkl", "wb") as f:
#             pickle.dump(self.embedding_matrix, f)

#     def load_preprocessing(self):
#         if not os.path.exists("word_index.pkl"):
#             return False

#         with open("word_index.pkl", "rb") as f:
#             self.word_index = pickle.load(f)

#         with open("embedding_matrix.pkl", "rb") as f:
#             self.embedding_matrix = pickle.load(f)

#         return True


#     def analyze_sentiment(self, text):
#         cleaned_tokens = self._clean_text(text)
#         sequence = self._texts_to_sequences([cleaned_tokens])

#         prediction = self.model.predict(sequence)[0]
#         print("Model Prediction (probabilities):", prediction)  # Debugging print statement
#         predicted_class = np.argmax(prediction)

#         sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
#         return sentiment_map[predicted_class], prediction




