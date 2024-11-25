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

        self.trend_weight = nn.Parameter(torch.Tensor(1))
        self.trend_bias = nn.Parameter(torch.Tensor(1))

        # Placeholder for periodic weights and bias; will be initialized in forward
        self.periodic_weight = None
        self.periodic_bias = None

        nn.init.uniform_(self.trend_weight)
        nn.init.uniform_(self.trend_bias)

    def forward(self, inputs):
        # Split inputs into x and t
        x = inputs[:, :, :self.num_vars-1]
        t = inputs[:, :, self.num_vars-1:]

        # Initialize periodic weights and bias based on input shape
        if self.periodic_weight is None:
            input_shape = inputs.shape
            self.periodic_weight = nn.Parameter(torch.Tensor(input_shape[-1] - self.num_vars + 1, self.num_frequency))
            self.periodic_bias = nn.Parameter(torch.Tensor(input_shape[1], self.num_frequency))
            nn.init.uniform_(self.periodic_weight)
            nn.init.uniform_(self.periodic_bias)
        
        # Trend component
        trend_component = self.trend_weight * t + self.trend_bias

        # Periodic component
        periodic_component = torch.sin(torch.matmul(t, self.periodic_weight) + self.periodic_bias)

        # Concatenate trend and periodic components
        t_encoded = torch.cat([trend_component, periodic_component], dim=-1)

        # Concatenate x and t_encoded
        output = torch.cat([x, t_encoded], dim=-1)
        
        return output