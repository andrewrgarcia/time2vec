# Keras Time2Vec

An intuitive, custom Keras Layer for Time2Vec Transformation.

Keras implementation of Non-local blocks from [[1]](https://arxiv.org/abs/1711.07971).


# Concept

Time2Vec offers a versatile representation of time with three fundamental properties. It encapsulates scalar notion of time $\tau$,  in $\mathbf{t2v}(\tau)$,
a vector of size k + 1. This transformation is defined as follows[[1]]:


```math
\mathbf{t2v}(\tau)[i] = 
    \begin{cases}
        \omega_i \tau + \phi_i \ifnum i = 0.\\
        \mathcal{F}(\omega_i \tau + \phi_i) \ifnum  \leq i \leq k
    \end{cases}
```



The below image shows a random walk and with its $\mathbf{t2v}$ (with a $k$ of k=20):
<center><img src="https://github.com/andrewrgarcia/keras-time2vec/blob/master/images/rep.png?raw=true" width=50% ></center>



# Usage Templates

The script `time2vec.py` contains the `Time2Vec` instance which takes a single or a group of time series and concatenates the above $\mathbf{t2v}$ tensor to it.

```python
from time2vec import Time2Vec

k = 20
time_series = Time2Vec(num_frequency=k)(time_series)
print(time_series)
...
```



1. Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker. "Time2Vec: Learning a Vector Representation of Time." arXiv:1907.05321 [cs.LG], 11 Jul 2019. [Link](https://arxiv.org/abs/1907.05321)