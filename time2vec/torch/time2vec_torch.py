import torch
import torch.nn as nn
import torch.nn.functional as F

class Time2Vec(nn.Module):
    def __init__(self, num_frequency, num_vars):
        """
        Custom PyTorch object for Time2Vec Transformation.

        Parameters
        -------------
        num_frequency: int
            The number of periodic components to model.
        num_vars: int
            The number of variables to consider from the input.
        """
        super(Time2Vec, self).__init__()
        self.num_frequency = num_frequency
        self.num_vars = num_vars

        # Initialize trend weights and biases
        self.trend_weight = nn.Parameter(torch.Tensor(1))
        self.trend_bias = nn.Parameter(torch.Tensor(1))

        # Periodic weights and biases will depend on input shape, initialized later
        self.periodic_weight = None
        self.periodic_bias = None

        # Initialize trend parameters
        nn.init.uniform_(self.trend_weight)
        nn.init.uniform_(self.trend_bias)

    def initialize_periodic_parameters(self, input_shape):
        """
        Initialize periodic weights and biases based on input shape.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input tensor (batch_size, sequence_length, num_features).
        """
        feature_dim = input_shape[-1] - self.num_vars + 1
        sequence_length = input_shape[1]

        # Define and register the parameters
        self.periodic_weight = nn.Parameter(torch.Tensor(feature_dim, self.num_frequency))
        self.periodic_bias = nn.Parameter(torch.Tensor(sequence_length, self.num_frequency))

        # Initialize with a smaller uniform range to match Keras
        nn.init.uniform_(self.periodic_weight, -0.05, 0.05)
        nn.init.uniform_(self.periodic_bias, -0.05, 0.05)


    def forward(self, inputs):
        """
        Forward pass of the Time2Vec layer.

        Parameters
        ----------
        inputs: torch.Tensor
            Input tensor of shape (batch_size, sequence_length, num_features).

        Returns
        -------
        torch.Tensor
            Output tensor with the trend and periodic components concatenated.
        """
        # Validate input shape
        if inputs.size(-1) < self.num_vars:
            raise ValueError(
                f"Input feature dimension ({inputs.size(-1)}) is less than the number of variables ({self.num_vars})."
            )

        # Split inputs into x (features) and t (time)
        x = inputs[:, :, :self.num_vars - 1]
        t = inputs[:, :, self.num_vars - 1:]

        # Initialize periodic parameters if not already initialized
        if self.periodic_weight is None or self.periodic_bias is None:
            self.initialize_periodic_parameters(inputs.shape)

        # Trend component
        trend_component = self.trend_weight * t + self.trend_bias

        # Periodic component
        periodic_component = torch.sin(torch.matmul(t, self.periodic_weight) + self.periodic_bias)

        # Concatenate trend and periodic components
        t_encoded = torch.cat([trend_component, periodic_component], dim=-1)

        # Concatenate x (features) and t_encoded (time encoding)
        output = torch.cat([x, t_encoded], dim=-1)

        return output

    def output_shape(self, input_shape):
        """
        Compute the output shape of the Time2Vec layer.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input tensor (batch_size, sequence_length, num_features).

        Returns
        -------
        tuple
            Output shape after the Time2Vec transformation.
        """
        batch_size, sequence_length, num_features = input_shape
        feature_dim = num_features - self.num_vars + 1
        output_feature_dim = feature_dim + self.num_frequency + self.num_vars - 1
        return (batch_size, sequence_length, output_feature_dim)
