# Time2Vec TensorFlow

An intuitive, custom TensorFlow.Keras Layer for Time2Vec Transformation.

Keras implementation of Non-local blocks from [[1]](https://arxiv.org/abs/1907.05321).


# Concept

Time2Vec offers a versatile representation of time with three fundamental properties. It encapsulates scalar notion of time $\tau$,  in $\mathbf{t2v}(\tau)$,
a vector of size k + 1. This transformation, for an $i^{th}$  element of $\mathbf{t2v}$, is defined as follows:


```math
\mathbf{t2v}(\tau)[i] = 
    \begin{cases}
        \omega_i \tau + \phi_i, & \mathrm{if} & i = 0.\\
        \mathcal{F}(\omega_i \tau + \phi_i), & \mathrm{if} & 1 \leq i \leq k.
    \end{cases}
```
The above incorporates a periodic activation function denoted as $\mathcal{F}$, and involves learnable parameters $\omega_i$ and $\phi_i$ [[1]](https://arxiv.org/abs/1907.05321). 

# Usage Templates

The script `time2vec.py` contains the `Time2Vec` instance which takes a single or a group of time series and concatenates the above $\mathbf{t2v}$ tensor to it.

```python
from time2vec import Time2Vec

k = 20
time_series = np.linspace(0, 100, 1000).reshape(-1,1)
time_series = Time2Vec(num_frequency=k)(time_series)
print(time_series)
...
```

For a more interactive experience check out this [Google Colab](https://colab.research.google.com/drive/1P2BOAaQlo54SqYCsL8FFq1PffDjQuO1F?usp=sharing)


1. Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker. "Time2Vec: Learning a Vector Representation of Time." arXiv:1907.05321 [cs.LG], 11 Jul 2019. [Link](https://arxiv.org/abs/1907.05321)
