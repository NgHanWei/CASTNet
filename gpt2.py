import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
# embed_dim = 32  # Embedding size for each token
# num_heads = 2  # Number of attention heads
# ff_dim = 32  # Hidden layer size in feed forward network inside transformer
# vocab_size = 20000  # Only consider the top 20k words
# maxlen = 16  # Only consider the first 200 words of each movie review

# inputs = layers.Input(shape=(maxlen,), dtype=tf.float64)
# embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
# x = embedding_layer(inputs)
# transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
# x = transformer_block(x)
# x = layers.GlobalAveragePooling1D()(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Dense(20, activation="relu")(x)
# x = layers.Dropout(0.1)(x)
# outputs = layers.Dense(2, activation="softmax")(x)

# model = keras.Model(inputs=inputs, outputs=outputs)

# print(model.summary())