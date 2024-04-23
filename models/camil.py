import os
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.callbacks import CallbackList
from models.custom_callbacks import CustomReduceLRoP
from tensorflow.keras.layers import Input, Dense, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from dataset_utils.camelyon_batch_generator import DataGenerator
from models.custom_layers import NeighborAggregator, CustomAttention, Last_Sigmoid, MILAttentionLayer, encoder
from flushed_print import print
import time
from collections import deque
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from models.metrics import eval_metric
from nystromformer.nystromformer import NystromAttention

class CAMIL:
    def __init__(self, args):
        """
        Build the architercure of the Graph Att net
        Parameters
        ----------
        arch:          a dict describing the architecture of the neural network to be trained
        mode            :str, specifying the version of the model (siamese, euclidean)
        containing input_types
        and input_placeholders for each key and value pair, respecively.
        input_shape:   tuple(int, int, int) of the input shape of the patches
        """
        self.input_shape = args.input_shape
        self.args = args
        self.n_classes=args.n_classes

        self.inputs = {
            'bag': Input(self.input_shape),
            'adjacency_matrix': Input(shape=(None, None), dtype='float32', name='adjacency_matrix', sparse=True)
        }


        self.subtyping = args.subtyping

        self.attcls = MILAttentionLayer(weight_params_dim=128, use_gated=True, kernel_regularizer=l2(1e-5, ))

        self.custom_att = CustomAttention(weight_params_dim=256, name="custom_att")
        self.wv = tf.keras.layers.Dense(512)

        self.neigh = NeighborAggregator(output_dim=1, name="alpha")

        self.nyst_att = NystromAttention(dim=512, dim_head=64, heads=8, num_landmarks=256,
                                         pinv_iterations=6)

        self.encoder = encoder()

        if self.subtyping:
                self.class_fc = Last_Sigmoid(output_dim=self.n_classes, name='FC1_sigmoid_1',
                                             kernel_regularizer=l2(1e-5),
                                             pooling_mode='sum', subtyping=self.subtyping)
        else:
                self.class_fc = Last_Sigmoid(output_dim=1, name='FC1_sigmoid_1',
                                             kernel_regularizer=l2(1e-5),
                                             pooling_mode='sum', subtyping=self.subtyping)



        xo, alpha = self.encoder([self.inputs['bag'], self.inputs['adjacency_matrix']])

        k_alpha = self.attcls(xo)
        attn_output = tf.keras.layers.multiply([k_alpha, xo])

        out = self.class_fc(attn_output)

        self.net = Model(inputs=[self.inputs['bag'], self.inputs["adjacency_matrix"]], outputs=[out, alpha, k_alpha])

    @property
    def model(self):
        return self.net

    def train(self, train_bags, fold, val_bags, args):
        """
        Train the Graph Att net
        Parameters
        ----------
        train_set       : a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches
        check_dir       :str, specifying directory where the weights of the siamese net are stored
        irun            :int, id of the experiment
        ifold           :int, fold of the k-corss fold validation
        weight_file     :boolen, specifying whether there is a weightflie or not
        Returns
        -------
        A History object containing  a record of models loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values.
        """

        train_gen = DataGenerator(args=args, batch_size=1, shuffle=False, filenames=train_bags,
                                  train=True)
        val_gen = DataGenerator(args=args, batch_size=1, shuffle=False, filenames=val_bags, train=True)

        if not os.path.exists(os.path.join(args.save_dir, fold)):
            os.makedirs(os.path.join(args.save_dir, fold, args.experiment_name), exist_ok=True)

        checkpoint_path = os.path.join(os.path.join(args.save_dir, fold, args.experiment_name),
                                       "{}.ckpt".format(args.experiment_name))

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         monitor='val_loss',
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         mode='auto',
                                                         save_freq='epoch',
                                                         verbose=1)

        _callbacks = [cp_callback]
        callbacks = CallbackList(_callbacks, add_history=True, model=self.net)

        logs = {}
        callbacks.on_train_begin(logs=logs)

        optimizer = Adam(lr=args.init_lr, beta_1=0.9, beta_2=0.999)

        reduce_rl_plateau = CustomReduceLRoP(patience=10,
                                             factor=0.2,
                                             verbose=1,
                                             optim_lr=optimizer.learning_rate,
                                             mode="min",
                                             reduce_lin=False)

        train_loss_tracker = tf.keras.metrics.Mean()
        val_loss_tracker = tf.keras.metrics.Mean()
        if not self.subtyping:
            loss_fn = BinaryCrossentropy(from_logits=False)
            train_acc_metric = tf.keras.metrics.BinaryAccuracy()
            val_acc_metric = tf.keras.metrics.BinaryAccuracy()
        else:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            train_acc_metric = tf.keras.metrics.Accuracy()
            val_acc_metric = tf.keras.metrics.Accuracy()

        @tf.function(experimental_relax_shapes=True)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits, scores, k_alpha = self.net(x, training=True)

                loss_value = loss_fn(y, logits)

            grads = tape.gradient(loss_value, self.net.trainable_weights)

            optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
            train_loss_tracker.update_state(loss_value)

            if not self.subtyping:
                train_acc_metric.update_state(y, logits)
            else:
                train_acc_metric.update_state(y, tf.argmax(logits, 1))

            return {"train_loss": train_loss_tracker.result(), "train_accuracy": train_acc_metric.result()}

        @tf.function(experimental_relax_shapes=True)
        def val_step(x, y):
            logits, scores, k_alpha = self.net(x, training=False)

            val_loss = loss_fn(y, logits)
            val_loss_tracker.update_state(val_loss)

            if not self.subtyping:
                val_acc_metric.update_state(y, logits)
            else:
                val_acc_metric.update_state(y, tf.argmax(logits, 1))
            return val_loss

        early_stopping = 20
        loss_history = deque(maxlen=early_stopping + 1)
        reduce_rl_plateau.on_train_begin()
        for epoch in range(args.epochs):

            train_loss_tracker.reset_states()
            train_acc_metric.reset_states()
            val_loss_tracker.reset_states()
            val_acc_metric.reset_states()

            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(train_gen):

                callbacks.on_batch_begin(step, logs=logs)
                callbacks.on_train_batch_begin(step, logs=logs)
                train_dict = train_step(x_batch_train, np.expand_dims(y_batch_train, axis=0))

                logs["train_loss"] = train_dict["train_loss"]

                callbacks.on_train_batch_end(step, logs=logs)
                callbacks.on_batch_end(step, logs=logs)
                if (step + 1) % 50 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(logs["train_loss"])))

            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            for step, (x_batch_val, y_batch_val) in enumerate(val_gen):
                callbacks.on_batch_begin(step, logs=logs)
                callbacks.on_test_batch_begin(step, logs=logs)
                val_step(x_batch_val, np.expand_dims(y_batch_val, axis=0))

                callbacks.on_test_batch_end(step, logs=logs)
                callbacks.on_batch_end(step, logs=logs)

            logs["val_loss"] = val_loss_tracker.result()

            loss_history.append(val_loss_tracker.result())
            val_acc = val_acc_metric.result()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))
            reduce_rl_plateau.on_epoch_end(epoch, val_loss_tracker.result())
            callbacks.on_epoch_end(epoch, logs=logs)

            if len(loss_history) > early_stopping:
                if loss_history.popleft() < min(loss_history):
                    print(f'\nEarly stopping. No validation loss '
                          f'improvement in {early_stopping} epochs.')
                    break

        callbacks.on_train_end(logs=logs)

    def predict(self, test_bags, fold, args, test_model):

        """
        Evaluate the transformer_k set
        Parameters
        ----------
        test_set: a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches
        Returns
        -------
        test_loss : float reffering to the transformer_k loss
        acc       : float reffering to the transformer_k accuracy
        precision : float reffering to the transformer_k precision
        recall    : float referring to the transformer_k recall
        auc       : float reffering to the transformer_k auc
        """


        if not self.subtyping:
            eval_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        else:
            eval_accuracy_metric = tf.keras.metrics.Accuracy()

        checkpoint_path = os.path.join(os.path.join(args.save_dir, fold, args.experiment_name),
                                       "{}.ckpt".format(args.experiment_name))
        test_model.load_weights(checkpoint_path)

        test_gen = DataGenerator(args=args, batch_size=1, filenames=test_bags, train=False)

        @tf.function(experimental_relax_shapes=True)
        def test_step(images, labels):
            pred, scores, k_alpha = test_model(images, training=False)
            if self.subtyping:
                eval_accuracy_metric.update_state(labels, tf.argmax(pred, 1))
            else:
                eval_accuracy_metric.update_state(labels, pred)
            return pred, k_alpha

        y_pred = []
        y_true = []
        os.makedirs(args.raw_save_dir, exist_ok=True)

        for enum, (x_batch_val, y_batch_val) in enumerate(test_gen):
            pred , k_alpha= test_step(x_batch_val, np.expand_dims(y_batch_val, axis=0))

            y_true.append(np.expand_dims(y_batch_val, axis=0))
            y_pred.append(pred.numpy().tolist()[0])


        if self.subtyping:

            y_pred = np.reshape(y_pred, (-1, 2))
            y_pred = y_pred[:, 1]
            #y_pred = np.reshape(y_pred, (-1, self.n_classes))
            # auc_0 = roc_auc_score(y_true, y_pred, average="macro", multi_class='ovr')

            macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0 = eval_metric(y_pred, y_true)
            print(macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0)

            auc_0 = roc_auc_score(y_true, y_pred, average="macro")
            print("AUC {}".format(auc_0))

            test_acc = eval_accuracy_metric.result()
            print("Test acc: %.4f" % (float(test_acc),))

            y_pred = np.argmax(y_pred, axis=1)
            mF1_0 = f1_score(y_true, y_pred, average='macro')
            print("Fscore: %.4f" % (float(mF1_0),))

            recall = recall_score(y_true, y_pred,average='macro')
            print("recall: %.4f" % (float(recall),))

            precision = precision_score(y_true, y_pred, average='macro')
            print("precision: %.4f" % (float(precision),))

        else:
            macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0 = eval_metric(y_pred, y_true)
            print(macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0)

            test_acc = eval_accuracy_metric.result()
            print("Test acc: %.4f" % (float(test_acc),))

            auc = roc_auc_score(y_true, y_pred)
            print("AUC {}".format(auc))

            y_pred = np.array(y_pred).ravel()
            y_pred = tf.round(y_pred)
            mF1_0 = f1_score(y_true, y_pred)
            print("Fscore: %.4f" % (float(mF1_0),))

            recall = recall_score(y_true, y_pred)
            print("recall: %.4f" % (float(recall),))

            precision = precision_score(y_true, y_pred)
            print("precision: %.4f" % (float(precision),))

        return test_acc, auc