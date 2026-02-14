import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle
import re


MAX_SEQUENCE_LENGTH = 35
EMBEDDING_DIM = 70 # From Word2Vec vector_size
NUM_HEADS = 3
FF_DIM = 120
NUM_CLASSES = 3 

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, token_initial_weights=None, **kwargs):
        super().__init__(**kwargs)
        if token_initial_weights is not None:
            # Initialize with pre-trained weights, and optionally make it non-trainable
            self.token_embeddings = layers.Embedding(
                input_dim=vocab_size,
                output_dim=embed_dim,
                mask_zero=True,
                weights=[token_initial_weights],
                trainable=False # Typically, pre-trained embeddings are not trained further, or at least not initially
            )
        else:
            self.token_embeddings = layers.Embedding(
                input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
            )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def build(self, input_shape):
        super().build(input_shape)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        # self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        # ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def build(self, input_shape):
        super().build(input_shape)



class SentimentAI:
    def __init__(self):
        self.model = self.load_model() 
        preprocessing_loaded = self.load_preprocessing()
        if self.model is None or not preprocessing_loaded:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('punkt_tab')
            train_df = pd.read_csv("train.csv", encoding="latin1")
            test_df = pd.read_csv("test.csv", encoding="latin1")

            self.stop_words = set(stopwords.words('english'))

            # Combine text for Word2Vec training
            all_raw_text = pd.concat([train_df['text'], test_df['text']], axis=0).astype(str).tolist()
            self.word2vec_model, self.word_index, self.embedding_matrix = self._train_word2vec_model(all_raw_text)
            self.save_preprocessing()

            self.train_data = self.preprocess_data(train_df)
            self.test_data = self.preprocess_data(test_df)

            self.model = self.train_model(self.train_data)
            self.save_model() 

    def _clean_text(self, text):
        tokens = word_tokenize(str(text))
        tokens = [word.lower() for word in tokens if word.isalpha()]
        # tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [word for word in tokens if len(word)>1]
        return tokens

    def _train_word2vec_model(self, all_raw_text):
        # Tokenize all text for Word2Vec
        tokenized_sentences = [self._clean_text(text) for text in all_raw_text]

        # Train Word2Vec model
        word2vec_model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=EMBEDDING_DIM,
            window=5,
            min_count=2,
            workers=4
        )

        # Create word_index and embedding_matrix
        word_index = {"PAD": 0} # 0 for padding
        for i, word in enumerate(word2vec_model.wv.index_to_key):
            word_index[word] = i + 1  # Start from 1

        vocab_size = len(word_index)
        embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
        for word, i in word_index.items():
            if word != "PAD": # Skip PAD, it's already 0
                embedding_matrix[i] = word2vec_model.wv[word]

        return word2vec_model, word_index, embedding_matrix


    def preprocess_data(self, data):
        data = data.drop(columns=[
            'textID', 'selected_text', 'Time of Tweet',
            'Age of User', 'Country', 'Population -2020',
            'Land Area (Km²)', 'Density (P/Km²)'
        ], errors='ignore')

        data["sentiment"] = data["sentiment"].map({
            'positive': 2, # Map to 0, 1, 2 for SparseCategoricalCrossentropy
            'neutral': 1,
            'negative': 0
        })

        data = data.dropna(subset=['text', 'sentiment']).reset_index(drop=True)

        # Apply cleaning to 'text' column, storing tokenized lists
        data["text"] = data["text"].apply(self._clean_text)

        return data

    def _texts_to_sequences(self, texts):
        sequences = []
        for text_tokens in texts:
            sequence = [self.word_index.get(word, 0) for word in text_tokens]
            sequences.append(sequence)
        return keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')


    def train_model(self, data):
        X = self._texts_to_sequences(data["text"])
        y = data["sentiment"].values

        inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH,))

        # Use Embedding layer initialized with Word2Vec weights passed during initialization
        embedding_layer = PositionalEmbedding(
            MAX_SEQUENCE_LENGTH,
            len(self.word_index),
            EMBEDDING_DIM,
            token_initial_weights=self.embedding_matrix
        )
        x = embedding_layer(inputs)

        # Transformer Blocks
        for _ in range(2): # Use 2 transformer blocks
            x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, FF_DIM)(x)

        x = layers.GlobalAveragePooling1D()(x) 
        # x = layers.Dropout(0.1)(x)
        x = layers.Dense(10, activation="relu")(x)
        # x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("Training Transformer model...")
        model.fit(
            X,
            y,
            batch_size=64,
            epochs=40, 
            validation_split=0.1
        )
        print("Transformer model trained.")
        return model
    
    def save_model(self, model_path="sentiment_transformer_model.keras"):
        self.model.save(model_path,include_optimizer=False)
    
    def load_model(self, model_path="sentiment_transformer_model.keras"):
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            print("No pre-trained model found. A new model will be trained.")
            return None

        try:
            return keras.models.load_model(
                model_path,
                custom_objects={
                    "PositionalEmbedding": PositionalEmbedding,
                    "TransformerBlock": TransformerBlock
                }
            )
        except Exception as e:
            print("Error loading model:", e)
            return None
        
    def save_preprocessing(self):
        with open("word_index.pkl", "wb") as f:
            pickle.dump(self.word_index, f)

        with open("embedding_matrix.pkl", "wb") as f:
            pickle.dump(self.embedding_matrix, f)

    def load_preprocessing(self):
        if not os.path.exists("word_index.pkl"):
            return False

        with open("word_index.pkl", "rb") as f:
            self.word_index = pickle.load(f)

        with open("embedding_matrix.pkl", "rb") as f:
            self.embedding_matrix = pickle.load(f)

        return True


    def analyze_sentiment(self, text):
        cleaned_tokens = self._clean_text(text)
        sequence = self._texts_to_sequences([cleaned_tokens])

        prediction = self.model.predict(sequence)[0]
        print("Model Prediction (probabilities):", prediction) 
        predicted_class = np.argmax(prediction)

        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return sentiment_map[predicted_class], prediction


 
def split_into_sentences(text):
    # Simple sentence splitter
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def analyze_sentiment(sentiment_ai, text):
    sentiment_result, scores = sentiment_ai.analyze_sentiment(text)
    print("Sentiment Result:", sentiment_result, "Scores:", scores) 
    return {
        "label": sentiment_result,
        "scores": {
        "Positive": scores[2],
        "Neutral": scores[1],
        "Negative": scores[0]
        }
    }

# -----------------------------
# Page Config
# -----------------------------

sentiment_ai = SentimentAI()  # Initialize model once at the start

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("Sentiment Analysis Dashboard")
    st.markdown("### Model Parameters")
    st.markdown("**Model Type:** Custom Transformer + Word2Vec")
    st.markdown("**Libraries:** TensorFlow, NLTK, Gensim")
    st.markdown("**Preprocessing:**")
    st.markdown("- Tokenization (NLTK)")
    st.markdown("- Lowercasing")
    st.markdown("- Remove Non-Alphabetic Tokens")
    st.markdown("- Remove Short Words")
    st.markdown("- Word2Vec Embeddings")
    st.markdown("- Padding & Sequence Encoding")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("## Analyze Text Sentiment")

col1, col2 = st.columns([5, 1])

with col1:
    text_input = st.text_input("Enter your text here...")

with col2:
    analyze_btn = st.button("Analyze Text")

st.markdown("**Or Upload Document**")

uploaded_file = st.file_uploader(
    "Choose .txt or .csv file",
    type=["txt", "csv"]
)

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")

    # Split into sentences
    sentences = split_into_sentences(content)
    if sentences:
        st.subheader("Sentence-wise Sentiment Analysis for uploaded document")
        results = []
        for sentence in sentences:
            sentiment, probabilities = sentiment_ai.analyze_sentiment(sentence)
            confidence = max(probabilities)
            results.append({
                "Sentence": sentence,
                "Predicted Sentiment": sentiment,
                "Confidence (%)": round(confidence * 100, 2)
            })
        # Display as table
        import pandas as pd
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
    else:
        st.warning("No valid sentences found in the uploaded file.")


# -----------------------------
# Output Section
# -----------------------------
if analyze_btn:
    if text_input.strip() == "":
        st.warning("Please enter text or upload a file.")
    else:
        result = analyze_sentiment(sentiment_ai,text_input)
        label = result["label"]
        scores = result["scores"]

        st.divider()

        # -----------------------------
        # Enterprise Color Map
        # -----------------------------
        color_map = {
            "Positive": "#2E7D32",   # deep green
            "Neutral":  "#F9A825",   # amber
            "Negative": "#C62828"    # dark red
        }

        # -----------------------------
        # Overall Sentiment Badge
        # -----------------------------
        st.markdown("### SENTIMENT ANALYSIS for Input Text")
        st.markdown(
            f"""
            <div style="
                display:inline-block;
                padding:6px 14px;
                border-radius:999px;
                font-weight:600;
                font-size:16px;
                background-color:{color_map[label]}22;
                color:{color_map[label]};
            ">
                {label}
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------
        # Sentiment Probability Chart
        # -----------------------------
        st.markdown("### Sentiment Probability Scores")

        sentiments = list(scores.keys())
        values = list(scores.values())
        colors = [color_map[s] for s in sentiments]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")

        bars = ax.barh(
            sentiments,
            values,
            color=colors,
            height=0.45
        )

        # Axes styling for dark theme
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", color="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Percentage labels
        for bar, value, sentiment in zip(bars, values, sentiments):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"{float(round(value, 3) * 100):.2f}%",
                va="center",
                ha="left",
                fontsize=11,
                fontweight="bold",
                color=color_map[sentiment]
            )

        st.pyplot(fig, use_container_width=True)

# -----------------------------
# About Section
# -----------------------------
st.divider()
st.markdown("### About analysis")
st.write(
    "Once analysis is finished, the overall sentiment and probability scores "
    "for positive, neutral, and negative sentiments are displayed. "
    "Better formatted input text improves accuracy and speed of analysis."
)