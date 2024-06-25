import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class Time2Vec(layers.Layer):
    def __init__(self, num_frequency, num_vars, **kwargs):
        """
        Custom Keras Layer for Time2Vec Transformation.

        Parameters
        -------------
        num_frequency: int
            The number of periodic components to model.
        num_vars: int
            The number of variables to consider from the input.
        """
        super(Time2Vec, self).__init__(**kwargs)
        self.num_frequency = num_frequency
        self.num_vars = num_vars

    def build(self, input_shape):
        super(Time2Vec, self).build(input_shape)

        self.trend_weight = self.add_weight(
            name="trend_weight",
            shape=(1,),
            initializer="uniform",
            trainable=True,
        )
        self.trend_bias = self.add_weight(
            name="trend_bias",
            shape=(1,),
            initializer="uniform",
            trainable=True,
        )

        self.periodic_weight = self.add_weight(
            name="periodic_weight",
            shape=(input_shape[-1] - self.num_vars + 1, self.num_frequency),
            initializer="uniform",
            trainable=True,
        )
        self.periodic_bias = self.add_weight(
            name="periodic_bias",
            shape=(input_shape[1], self.num_frequency),
            initializer="uniform",
            trainable=True,
        )

    def call(self, inputs):
        # Split inputs into x and t
        x = inputs[:, :, :self.num_vars-1]
        t = inputs[:, :, self.num_vars-1:]
        # print("\nt",t.shape)

        # Trend component
        trend_component = self.trend_weight * t + self.trend_bias

        # Periodic component
        periodic_component = tf.sin(tf.matmul(t, self.periodic_weight) + self.periodic_bias)

        # Concatenate trend and periodic components
        t_encoded = tf.concat([trend_component, periodic_component], axis=-1)

        # Concatenate x and t_encoded
        output = tf.concat([x, t_encoded], axis=-1)
        
        return output
    
    def obtain_output_shape(self, input_shape):
        """
        Obtain the output shape of the layer.

        Parameters
        ----------
        input_shape: tuple
            The shape of the input data.
        """
        # Sum of trend (1) and periodic components
        output_feature_dim = (input_shape[-1] - self.num_vars + 1) + self.num_frequency + self.num_vars - 1
        return (input_shape[0], input_shape[1], output_feature_dim)