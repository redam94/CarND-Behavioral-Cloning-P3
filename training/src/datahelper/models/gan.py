import tensorflow as tf
from tensorflow import keras, image
from tensorflow.keras import Model, layers, activations as acts, losses, metrics
from tensorflow.keras.layers import Layer
import numpy as np
from typing import Optional, Union
import time

class Patches(Layer):
    """Patch dim must be of length 4 [1, x_dim, y_dim, 1] or 2 [x_dim, y_dim] 
        modified from https://keras.io/examples/vision/image_classification_with_vision_transformer/"""
    def __init__(self, patch_dim: list[int]) -> None:
        super().__init__()
        if len(patch_dim) == 2:
            patch_dim = [1, patch_dim[0], patch_dim[1], 1]
        if len(patch_dim) != 4:
            raise ValueError
        self.patch_dim = patch_dim
    
    def call(self, images: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        batch_size = tf.shape(images)[0]
        patches = image.extract_patches(
            images = images,
            sizes = self.patch_dim,
            strides = self.patch_dim,
            rates = [1, 1, 1, 1],
            padding = "VALID" )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class EncoderModel(Model):
    def __init__(self, image_shape: tuple[int], in_channels: int, output_dim: int) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.conv3x3_layer1 = layers.Conv2D(32, (3, 3), strides=(2, 2), activation=acts.swish, name= "Conv3x3_layer_1")
        self.batch_normalization_layer1 = layers.BatchNormalization()
        self.conv3x3_layer2 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation=acts.swish, name = "Conv3x3_layer_2")
        self.conv3x3_layer3 = layers.Conv2D(128, (3, 3), strides=(2, 2), activation=acts.swish, name = "Conv3x3_layer_3")
        self.global_max = layers.GlobalMaxPooling2D()
        self.embedding = layers.Dense(output_dim, activation=acts.swish, name="output")

    def call(self, inputs : tf.Tensor) -> tf.Tensor:
        x = inputs
        x = self.conv3x3_layer1(x)
        x = self.batch_normalization_layer1(x)
        x = self.conv3x3_layer2(x)
        x = self.conv3x3_layer3(x)
        x = self.global_max(x)
        out = self.embedding(x)
        
        return out


class DecoderModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = layers.Dense(128, activation=acts.swish)
        self.reshape = layers.Reshape((1, 1, 128))
        self.conv2dTrans_1 = layers.Conv2DTranspose(128, (1, 4), strides=(1, 1), activation=acts.swish)
        self.conv2dTrans_2 = layers.Conv2DTranspose(64, (3, 3), strides=2, activation=acts.swish)
        self.conv2dTrans_3 = layers.Conv2DTranspose(32, (5, 3), strides=2, activation=acts.swish)
        self.conv2dTrans_4 = layers.Conv2DTranspose(16, (5, 5), strides=2, activation=acts.swish)
        self.out_img = layers.Conv2D(9, (2, 2), activation = 'sigmoid')

         

    def call(self, input : tf.Tensor) -> tf.Tensor:
        x = self.fc1(input)
        x = self.reshape(x)
        x = self.conv2dTrans_1(x)
        x = self.conv2dTrans_2(x)
        x = self.conv2dTrans_3(x)
        x = self.conv2dTrans_4(x)
        out_img = self.out_img(x)
        return out_img


class GeneratorModel(Model):
    def __init__(self, encoder_model: EncoderModel, decoder_model: DecoderModel) -> None:
        super().__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
    
    def call(self, inputs):
        encoding = self.encoder(inputs)
        img_gen = self.decoder(encoding)
        return img_gen, encoding

class AdversarialModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.conv3x3_layer1 = layers.Conv2D(32, (3, 3), activation=acts.swish, name= "Conv3x3_layer_1")
        self.batch_normalization_layer1 = layers.BatchNormalization()
        self.max_pooling_layer1 = layers.MaxPool2D()
        self.conv3x3_layer2 = layers.Conv2D(64, (3, 3), activation=acts.swish, name = "Conv3x3_layer_2")
        self.max_pooling_layer2 = layers.MaxPool2D()
        self.conv3x3_layer3 = layers.Conv2D(128, (3, 3), activation=acts.swish, name = "Conv3x3_layer_3")
        self.global_max = layers.GlobalMaxPooling2D()
        #self.concatinate = layers.Concatenate()
        self.fc1 = layers.Dense(32, activation=acts.swish, name="fc1")
        self.out = layers.Dense(1, name='is_real')
    def call(self, images, encoding):
        x = self.conv3x3_layer1(images)
        x = self.batch_normalization_layer1(x)
        x = self.max_pooling_layer1(x)
        x = self.conv3x3_layer2(x)
        x = self.max_pooling_layer2(x)
        x = self.conv3x3_layer3(x)
        x = self.global_max(x)
        #x = self.concatinate([x, encoding])
        x = self.fc1(x)
        out = self.out(x)
        return out

mse = losses.MeanSquaredError()
cross_entropy = losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def image_loss(images, gen_images, l=1e-6):
    return mse(images, gen_images) + l*tf.reduce_sum(image.total_variation(gen_images))

def adversarial_loss(real_output, fake_output):
    
    fake_loss = cross_entropy(tf.zeros_like(fake_output)+.1*tf.random.uniform(shape=fake_output.shape), fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output)*.9 + .1*tf.random.uniform(shape=real_output.shape), real_output)
    
    return fake_loss + real_loss


gen_loss_tracker = metrics.Mean(name="gen_loss")
adv_loss_tracker = metrics.Mean(name="adv_loss")
img_loss_tracker = metrics.Mean(name="img_loss")

class GAN(Model):
    def __init__(self, generator : GeneratorModel, adversary : AdversarialModel):
        super().__init__()
        self.generator = generator
        self.adversary = adversary
        
    def compile(self, adv_optimizer, gen_optimizer):
        super().compile()
        self.adv_optimizer = adv_optimizer
        self.gen_optimizer = gen_optimizer
        
    def call(self, x, training=True):
        img_gen, encoding = self.generator(x)
        return img_gen
    
    @tf.function() 
    def train_step(self, images):
        images, _ = images
        with tf.GradientTape() as gen_tape, tf.GradientTape() as adv_tape:
            generated_images, encoding = self.generator(images)
            
            real_output = self.adversary(images, encoding, training=True)
            fake_output = self.adversary(generated_images, encoding, training=True)

            gen_loss = generator_loss(fake_output)
            im_loss = image_loss(images, generated_images)
            adv_loss = adversarial_loss(real_output, fake_output)
            total_gen_loss = gen_loss + 25*im_loss
            

        gradients_of_generator = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        gradients_of_adversary = adv_tape.gradient(adv_loss, self.adversary.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.adv_optimizer.apply_gradients(zip(gradients_of_adversary, self.adversary.trainable_variables))
        gen_loss_tracker.update_state(total_gen_loss)
        adv_loss_tracker.update_state(adv_loss)
        img_loss_tracker.update_state(im_loss)
        return {'gen_loss': gen_loss_tracker.result(), 'adv_loss':adv_loss_tracker.result(), 'img_loss': img_loss_tracker.result()}

    
    #def train(self, dataset, epochs, batch_size):
    #    for epoch in range(epochs):
    #        start = time.time()
    #        dataset_len = dataset.shape[0]
    #        batches = dataset_len//batch_size
    #        for batch_number in range(batches):
    #            img_batch = dataset[batch_size*batch_number:batch_size*(batch_number+1)]
    #            self.train_step(img_batch)
    #        print(f"Time for epoch {epoch} is {time.time()-start} sec")




class FeatureExtractor(Model):
    """Demensional Reduction"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class Generator(Model):
    """Generates Next Frame"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class Discriminator(Model):
    """Discriminates between generative data and 'real' data"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class MultiHeadAttention(Model):
    """Multiheaded attention model"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class FullModel(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention_layer = MultiHeadAttention()
        self.feature_extractor = FeatureExtractor()
        self.generator = Generator()
        self.discriminator = Discriminator()
    