import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow import image
import numpy as np




class FeatureExtractor(Model):
    """Demensional Reduction"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Generator(Model):
    """Generates Next Frame"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Discriminator(Model):
    """Discriminates between generative data and 'real' data"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MultiHeadAttention(Model):
    """Multiheaded attention model"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class FullModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_layer = MultiHeadAttention()
        self.feature_extractor = FeatureExtractor()
        self.generator = Generator()
        self.discriminator = Discriminator()
    