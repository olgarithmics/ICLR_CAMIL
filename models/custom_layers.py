import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Dense, multiply
from nystromformer.nystromformer import NystromAttention
class MILAttentionLayer(tf.keras.layers.Layer):
    """Implementation of the attention-based Deep MIL layer.
    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.
    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
            self,
            weight_params_dim,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            use_gated=False,
            **kwargs,
    ):

        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = self.compute_attention_scores(inputs)

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=0)
        return alpha

    def compute_attention_scores(self, instance):

        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:
            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return tf.tensordot(instance, self.w_weight_params, axes=1)


class NeighborAggregator(tf.keras.layers.Layer):
    """
    Aggregation of neighborhood information
    This layer is responsible for aggregatting the neighborhood information of the attentin matrix through the
    element-wise multiplication with an adjacency matrix. Every row of the produced
    matrix is averaged to produce a single attention score.
    # Arguments
        output_dim:            positive integer, dimensionality of the output space
    # Input shape
        2D tensor with shape: (n, n)
        2d tensor with shape: (None, None) correspoding to the adjacency matrix
    # Output shape
        2D tensor with shape: (1, units) corresponding to the attention coefficients of every instance in the bag
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim

        super(NeighborAggregator, self).__init__(**kwargs)

    def call(self, inputs):
        data_input = inputs[0]

        adj_matrix = inputs[1]

        sparse_data_input = adj_matrix.__mul__(data_input)

        reduced_sum = tf.sparse.reduce_sum(sparse_data_input, 1)

        # reduced_mean = tf.math.divide(reduced_sum, self.k+1)
        # sparse_mean = tf.sparse.reduce_sum(sparse_data_input, 1)
        A_raw = tf.reshape(tensor=reduced_sum, shape=(tf.shape(data_input)[1],))

        alpha = K.softmax(A_raw)
        return alpha, A_raw

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)


class Last_Sigmoid(tf.keras.layers.Layer):
    """
    Attention Activation
    This layer contains the last sigmoid layer of the network
    # Arguments
        output_dim:         positive integer, dimensionality of the output space
        kernel_initializer: initializer of the `kernel` weights matrix
        bias_initializer:   initializer of the `bias` weights
        kernel_regularizer: regularizer function applied to the `kernel` weights matrix
        bias_regularizer:   regularizer function applied to the `bias` weights
        use_bias:           boolean, whether use bias or not
    # Input shape
        2D tensor with shape: (n, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, output_dim, subtyping,kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 pooling_mode="sum",
                 kernel_regularizer=None, bias_regularizer=None,norm=False,
                 use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.pooling_mode = pooling_mode
        self.use_bias = use_bias
        self.subtyping=subtyping


        super(Last_Sigmoid, self).__init__(**kwargs)

    def max_pooling(self,x):

        output = K.max(x, axis=0, keepdims=True)
        return output

    def sum_pooling(self,x):

        output =  K.sum(x, axis=0, keepdims=True)
        return output


    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]


        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x):

        if self.pooling_mode == 'max':
            x= self.max_pooling(x)
        if self.pooling_mode == 'sum':
            x= self.sum_pooling(x)

        if self.subtyping:
            x = K.dot(x, self.kernel)
            if self.use_bias:
                x = K.bias_add(x, self.bias)
            out = K.softmax(x)
        else:
            x = K.dot(x, self.kernel)
            if self.use_bias:
                x = K.bias_add(x, self.bias)
            out = K.sigmoid(x)
        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)


class CustomAttention(tf.keras.layers.Layer):

    def __init__(
            self,
            weight_params_dim,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.wq_init = self.kernel_initializer
        self.wk_init = self.kernel_initializer

        self.wq_regularizer = self.kernel_regularizer
        self.wk_regularizer = self.kernel_regularizer

    def build(self, input_shape):
        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[1]

        self.wq_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.wq_init,
            name="wq",
            regularizer=self.wq_regularizer,
            trainable=True,
        )

        self.wk_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.wk_init,
            name="wk",
            regularizer=self.wk_regularizer,
            trainable=True,
        )

        self.input_built = True

    def call(self, inputs):
        wsi_bag = inputs

        attention_weights = self.compute_attention_scores(wsi_bag)

        return attention_weights

    def compute_attention_scores(self, instance):
        q = tf.tensordot(instance, self.wq_weight_params, axes=1)

        k = tf.tensordot(instance, self.wk_weight_params, axes=1)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)

        matmul_qk = tf.tensordot(q, tf.transpose(k), axes=1)  # (..., seq_len_q, seq_len_k)

        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)


        return scaled_attention_logits


class encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(encoder, self).__init__()
        self.custom_att = CustomAttention(weight_params_dim=256, name="custom_att")
        self.wv = tf.keras.layers.Dense(512)

        self.neigh = NeighborAggregator(output_dim=1, name="alpha")

        self.nyst_att = NystromAttention(dim=512, dim_head=64, heads=8, num_landmarks=256,
                                         pinv_iterations=6)

    def call(self, inputs):

        dense = inputs[0]
        sparse_adj = inputs[1]

        encoder_output = self.nyst_att(tf.expand_dims(dense, axis=0), return_attn=False)
        xg = tf.ensure_shape(tf.squeeze(encoder_output), [None, 512])

        encoder_output = xg + dense

        attention_matrix = self.custom_att(encoder_output)
        norm_alpha, alpha = self.neigh([attention_matrix, sparse_adj])
        value = self.wv(dense)
        xl = multiply([norm_alpha, value], name="mul_1")

        wei = tf.math.sigmoid(-xl)
        squared_wei = tf.square(wei)
        xo = (xl * 2 * squared_wei) + 2* encoder_output * (1 - squared_wei)
        return xo, alpha

