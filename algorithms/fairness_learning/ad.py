import numpy as np
import scipy.special
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
import copy
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
try:
    import tensorflow.compat.v1 as tf
except ImportError as error:
    from logging import warning
    warning("{}: AdversarialDebiasing will be unavailable. To install, run:\n"
            "pip install 'aif360[AdversarialDebiasing]'".format(error))

from aif360.sklearn.utils import check_inputs, check_groups


class AdversarialDebiasing:
    """Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [5]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.

    References:
        .. [5] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
           Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
           Intelligence, Ethics, and Society, 2018.
    """

    def __init__(self,
                 scope_name,
                 sess,
                 seed=None,
                 adversary_loss_weight=0.1,
                 num_epochs=50,
                 batch_size=128,
                 classifier_num_hidden_units=200,
                 debias=True):
        """
        Args:
            unprivileged_groups (tuple): Representation for unprivileged groups
            privileged_groups (tuple): Representation for privileged groups
            scope_name (str): scope name for the tenforflow variables
            sess (tf.Session): tensorflow session
            seed (int, optional): Seed to make `predict` repeatable.
            adversary_loss_weight (float, optional): Hyperparameter that chooses
                the strength of the adversarial loss.
            num_epochs (int, optional): Number of training epochs.
            batch_size (int, optional): Batch size.
            classifier_num_hidden_units (int, optional): Number of hidden units
                in the classifier model.
            debias (bool, optional): Learn a classifier with or without
                debiasing.
        """
        self.scope_name = scope_name
        self.seed = seed

        self.sess = sess
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias

        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = models.resnet50(pretrained=True).to(self.device)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()

    def _classifier_model(self, features, features_dim, keep_prob):
        """Compute the classifier predictions for the outcome variable.
        """
        with tf.variable_scope("classifier_model"):
            W1 = tf.get_variable('W1', [features_dim, self.classifier_num_hidden_units],
                                  initializer=tf.initializers.glorot_uniform(seed=self.seed1))
            b1 = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units]), name='b1')

            h1 = tf.nn.relu(tf.matmul(features, W1) + b1)
            h1 = tf.nn.dropout(h1, keep_prob=keep_prob, seed=self.seed2)

            W2 = tf.get_variable('W2', [self.classifier_num_hidden_units, 1],
                                 initializer=tf.initializers.glorot_uniform(seed=self.seed3))
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_logit = tf.matmul(h1, W2) + b2
            pred_label = tf.sigmoid(pred_logit)

        return pred_label, pred_logit

    def _adversary_model(self, pred_logits, true_labels):
        """Compute the adversary predictions for the protected attribute.
        """
        with tf.variable_scope("adversary_model"):
            c = tf.get_variable('c', initializer=tf.constant(1.0))
            s = tf.sigmoid((1 + tf.abs(c)) * pred_logits)

            W2 = tf.get_variable('W2', [3, 1],
                                 initializer=tf.initializers.glorot_uniform(seed=self.seed4))
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_protected_attribute_logit = tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1), W2) + b2
            pred_protected_attribute_label = tf.sigmoid(pred_protected_attribute_logit)

        return pred_protected_attribute_label, pred_protected_attribute_logit

    def fit(self, dataset):
        """Compute the model parameters of the fair classifier using gradient
        descent.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            AdversarialDebiasing: Returns self.
        """
        if tf.executing_eagerly():
            raise RuntimeError("AdversarialDebiasing does not work in eager "
                    "execution mode. To fix, add `tf.disable_eager_execution()`"
                    " to the top of the calling script.")

        if self.seed is not None:
            np.random.seed(self.seed)
        ii32 = np.iinfo(np.int32)
        self.seed1, self.seed2, self.seed3, self.seed4 = np.random.randint(ii32.min, ii32.max, size=4)


        with tf.variable_scope(self.scope_name):
            num_train_samples = len(dataset.labels)
            self.features_dim = 2048

            # Setup placeholders
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.keep_prob = tf.placeholder(tf.float32)

            # Obtain classifier predictions and classifier loss
            self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
            pred_labels_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

            if self.debias:
                # Obtain adversary predictions and adversary loss
                pred_protected_attributes_labels, pred_protected_attributes_logits = self._adversary_model(pred_logits, self.true_labels_ph)
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph, logits=pred_protected_attributes_logits))

            # Setup optimizers with learning rates
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       1000, 0.96, staircase=True)
            classifier_opt = tf.train.AdamOptimizer(learning_rate)
            if self.debias:
                adversary_opt = tf.train.AdamOptimizer(learning_rate)

            classifier_vars = [var for var in tf.trainable_variables(scope=self.scope_name) if 'classifier_model' in var.name]
            if self.debias:
                adversary_vars = [var for var in tf.trainable_variables(scope=self.scope_name) if 'adversary_model' in var.name]
                # Update classifier parameters
                adversary_grads = {var: grad for (grad, var) in adversary_opt.compute_gradients(pred_protected_attributes_loss,
                                                                                      var_list=classifier_vars)}
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            classifier_grads = []
            for (grad,var) in classifier_opt.compute_gradients(pred_labels_loss, var_list=classifier_vars):
                if self.debias:
                    unit_adversary_grad = normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[var]
                classifier_grads.append((grad, var))
            classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

            if self.debias:
                # Update adversary parameters
                with tf.control_dependencies([classifier_minimizer]):
                    adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss, var_list=adversary_vars)#, global_step=global_step)

            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            data = dataset.data
            batch_size = 256
            data = torch.tensor(data).permute(0, 3, 1, 2).float()
            data = TensorDataset(data)
            data = DataLoader(data, batch_size=batch_size, shuffle=False)
            features_list = []
            with torch.no_grad():
                for batch in data:
                    batch = batch[0].to(self.device) 
                    features = self.resnet50(batch) 
                    features_list.append(features.squeeze().cpu()) 
            X = torch.cat(features_list, dim=0).numpy()

            Y = dataset.labels
            A = dataset.sensitive_attribute
            
            # Begin training
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples//self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size*i: self.batch_size*(i+1)]
                    data = dataset.data
                    

                    batch_features = X[batch_ids]
                    batch_labels = Y[batch_ids].reshape(-1, 1)
                    batch_protected_attributes = A[batch_ids].reshape(-1, 1)

                    batch_feed_dict = {self.features_ph: batch_features,
                                       self.true_labels_ph: batch_labels,
                                       self.protected_attributes_ph: batch_protected_attributes,
                                       self.keep_prob: 0.8}
                    if self.debias:
                        _, _, pred_labels_loss_value, pred_protected_attributes_loss_vale = self.sess.run([classifier_minimizer,
                                       adversary_minimizer,
                                       pred_labels_loss,
                                       pred_protected_attributes_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (epoch, i, pred_labels_loss_value,
                                                                                     pred_protected_attributes_loss_vale))
                    else:
                        _, pred_labels_loss_value = self.sess.run(
                            [classifier_minimizer,
                             pred_labels_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, pred_labels_loss_value))
        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the fair
        classifier learned.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        num_test_samples = len(dataset.labels)

        samples_covered = 0
        pred_labels = []

        data = dataset.data
        batch_size = 256
        data = torch.tensor(data).permute(0, 3, 1, 2).float()
        data = TensorDataset(data)
        data = DataLoader(data, batch_size=batch_size, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in data:
                batch = batch[0].to(self.device) 
                features = self.resnet50(batch) 
                features_list.append(features.squeeze().cpu()) 
        X = torch.cat(features_list, dim=0).numpy()

        Y = dataset.labels
        A = dataset.sensitive_attribute
        
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = X[batch_ids]
            batch_labels = Y[batch_ids].reshape(-1, 1)
            batch_protected_attributes = A[batch_ids].reshape(-1, 1)

            batch_feed_dict = {self.features_ph: batch_features,
                               self.true_labels_ph: batch_labels,
                               self.protected_attributes_ph: batch_protected_attributes,
                               self.keep_prob: 1.0}

            pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict)[:,0].tolist()
            samples_covered += len(batch_features)

        # Mutated, fairer dataset with new labels
        dataset_new = copy.deepcopy(dataset)
        dataset_new.scores = np.array(pred_labels, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(pred_labels)>0.5).astype(np.float64).reshape(-1,1)


        return dataset_new

# class AdversarialDebiasing(BaseEstimator, ClassifierMixin):
#     """Debiasing with adversarial learning.

#     Adversarial debiasing is an in-processing technique that learns a
#     classifier to maximize prediction accuracy and simultaneously reduce an
#     adversary's ability to determine the protected attribute from the
#     predictions [#zhang18]_. This approach leads to a fair classifier as the
#     predictions cannot carry any group discrimination information that the
#     adversary can exploit.

#     References:
#         .. [#zhang18] `B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating
#            Unwanted Biases with Adversarial Learning," AAAI/ACM Conference on
#            Artificial Intelligence, Ethics, and Society, 2018.
#            <https://dl.acm.org/citation.cfm?id=3278779>`_

#     Attributes:
#         groups_ (array, shape (n_groups,)): A list of group labels known to the
#             classifier.
#         classes_ (array, shape (n_classes,)): A list of class labels known to
#             the classifier.
#         sess_ (tensorflow.Session): The TensorFlow Session used for the
#             computations. Note: this can be manually closed to free up resources
#             with `self.sess_.close()`.
#         classifier_logits_ (tensorflow.Tensor): Tensor containing output logits
#             from the classifier.
#         adversary_logits_ (tensorflow.Tensor): Tensor containing output logits
#             from the adversary.
#     """

#     def __init__(self, scope_name='classifier',
#                  adversary_loss_weight=0.1, num_epochs=50, batch_size=128,
#                  classifier_num_hidden_units=200, debias=True, verbose=False,
#                  random_state=None):
#         r"""
#         Args:
#             prot_attr (single label or list-like, optional): Protected
#                 attribute(s) to use in the debiasing process. If more than one
#                 attribute, all combinations of values (intersections) are
#                 considered. Default is ``None`` meaning all protected attributes
#                 from the dataset are used.
#             scope_name (str, optional): TensorFlow "variable_scope" name for the
#                 entire model (classifier and adversary).
#             adversary_loss_weight (float or ``None``, optional): If ``None``,
#                 this will use the suggestion from the paper:
#                 :math:`\alpha = \sqrt{global\_step}` with inverse time decay on
#                 the learning rate. Otherwise, it uses the provided coefficient
#                 with exponential learning rate decay.
#             num_epochs (int, optional): Number of epochs for which to train.
#             batch_size (int, optional): Size of mini-batch for training.
#             classifier_num_hidden_units (int, optional): Number of hidden units
#                 in the classifier.
#             debias (bool, optional): If ``False``, learn a classifier without an
#                 adversary.
#             verbose (bool, optional): If ``True``, print losses every 200 steps.
#             random_state (int or numpy.RandomState, optional): Seed of pseudo-
#                 random number generator for shuffling data and seeding weights.
#         """

#         self.scope_name = scope_name
#         self.adversary_loss_weight = adversary_loss_weight
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.classifier_num_hidden_units = classifier_num_hidden_units
#         self.debias = debias
#         self.verbose = verbose
#         self.random_state = random_state

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.resnet50 = models.resnet50(pretrained=True).to(self.device)
#         self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
#         self.resnet50.eval()

#     def fit(self, dataset):
#         """Train the classifier and adversary (if ``debias == True``) with the
#         given training data.

#         Args:
#             X (pandas.DataFrame): Training samples.
#             y (array-like): Training labels.

#         Returns:
#             self
#         """
#         if tf.executing_eagerly():
#             raise RuntimeError("AdversarialDebiasing does not work in eager "
#                     "execution mode. To fix, add `tf.disable_eager_execution()`"
#                     " to the top of the calling script.")

#         data = dataset.data
#         batch_size = 256
#         data = torch.tensor(data).permute(0, 3, 1, 2).float()
#         data = TensorDataset(data)
#         data = DataLoader(data, batch_size=batch_size, shuffle=False)
#         features_list = []
#         with torch.no_grad():
#             for batch in data:
#                 batch = batch[0].to(self.device) 
#                 features = self.resnet50(batch) 
#                 features_list.append(features.squeeze().cpu()) 
#         X = torch.cat(features_list, dim=0).numpy()

#         Y = dataset.labels
#         A = dataset.sensitive_attribute
        
#         X, y, _ = check_inputs(X, y)
#         rng = check_random_state(self.random_state)
#         ii32 = np.iinfo(np.int32)
#         s1, s2, s3, s4 = rng.randint(ii32.min, ii32.max, size=4)

#         tf.reset_default_graph()
#         self.sess_ = tf.Session()

#         le = LabelEncoder()
#         y = le.fit_transform(y)
#         self.classes_ = le.classes_
#         # BUG: LabelEncoder converts to ndarray which removes tuple formatting
#         self.groups_ = le.classes_

#         n_classes = len(self.classes_)
#         n_groups = len(self.groups_)
#         # use sigmoid for binary case
#         if n_classes == 2:
#             n_classes = 1
#         if n_groups == 2:
#             n_groups = 1

#         n_samples, n_features = X.shape

#         with tf.variable_scope(self.scope_name):
#             # Setup placeholders
#             self.input_ph = tf.placeholder(tf.float32, shape=[None, n_features])
#             self.prot_attr_ph = tf.placeholder(tf.float32, shape=[None, 1])
#             self.true_labels_ph = tf.placeholder(tf.float32, shape=[None, 1])
#             self.keep_prob = tf.placeholder(tf.float32)

#             # Create classifier
#             with tf.variable_scope('classifier_model'):
#                 W1 = tf.get_variable(
#                         'W1', [n_features, self.classifier_num_hidden_units],
#                         initializer=tf.initializers.glorot_uniform(seed=s1))
#                 b1 = tf.Variable(tf.zeros(
#                         shape=[self.classifier_num_hidden_units]), name='b1')

#                 h1 = tf.nn.relu(tf.matmul(self.input_ph, W1) + b1)
#                 h1 = tf.nn.dropout(h1, rate=1-self.keep_prob, seed=s2)

#                 W2 = tf.get_variable(
#                         'W2', [self.classifier_num_hidden_units, n_classes],
#                         initializer=tf.initializers.glorot_uniform(seed=s3))
#                 b2 = tf.Variable(tf.zeros(shape=[n_classes]), name='b2')

#                 self.classifier_logits_ = tf.matmul(h1, W2) + b2

#             # Obtain classifier loss
#             if self.classifier_logits_.shape[1] == 1:
#                 clf_loss = tf.reduce_mean(
#                         tf.nn.sigmoid_cross_entropy_with_logits(
#                                 labels=self.true_labels_ph,
#                                 logits=self.classifier_logits_))
#             else:
#                 clf_loss = tf.reduce_mean(
#                         tf.nn.sparse_softmax_cross_entropy_with_logits(
#                                 labels=tf.squeeze(tf.cast(self.true_labels_ph,
#                                                           tf.int32)),
#                                 logits=self.classifier_logits_))

#             if self.debias:
#                 # Create adversary
#                 with tf.variable_scope("adversary_model"):
#                     c = tf.get_variable('c', initializer=tf.constant(1.0))
#                     s = tf.sigmoid((1 + tf.abs(c)) * self.classifier_logits_)

#                     W2 = tf.get_variable('W2', [3, n_groups],
#                             initializer=tf.initializers.glorot_uniform(seed=s4))
#                     b2 = tf.Variable(tf.zeros(shape=[n_groups]), name='b2')

#                     self.adversary_logits_ = tf.matmul(
#                             tf.concat([s, s * self.true_labels_ph,
#                                        s * (1. - self.true_labels_ph)], axis=1),
#                             W2) + b2

#                 # Obtain adversary loss
#                 if self.adversary_logits_.shape[1] == 1:
#                     adv_loss = tf.reduce_mean(
#                             tf.nn.sigmoid_cross_entropy_with_logits(
#                                     labels=self.prot_attr_ph,
#                                     logits=self.adversary_logits_))
#                 else:
#                     adv_loss = tf.reduce_mean(
#                             tf.nn.sparse_softmax_cross_entropy_with_logits(
#                                     labels=tf.squeeze(tf.cast(self.prot_attr_ph,
#                                                               tf.int32)),
#                                     logits=self.adversary_logits_))

#             global_step = tf.Variable(0., trainable=False)
#             init_learning_rate = 0.001
#             if self.adversary_loss_weight is not None:
#                 learning_rate = tf.train.exponential_decay(init_learning_rate,
#                     global_step, 1000, 0.96, staircase=True)
#             else:
#                 learning_rate = tf.train.inverse_time_decay(init_learning_rate,
#                         global_step, 1000, 0.1, staircase=True)

#             # Setup optimizers
#             clf_opt = tf.train.AdamOptimizer(learning_rate)
#             if self.debias:
#                 adv_opt = tf.train.AdamOptimizer(learning_rate)

#             clf_vars = [var for var in tf.trainable_variables()
#                         if 'classifier_model' in var.name]
#             if self.debias:
#                 adv_vars = [var for var in tf.trainable_variables()
#                             if 'adversary_model' in var.name]
#                 # Compute grad wrt classifier parameters
#                 adv_grads = {var: grad for (grad, var) in
#                         adv_opt.compute_gradients(adv_loss, var_list=clf_vars)}

#             normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

#             clf_grads = []
#             for (grad, var) in clf_opt.compute_gradients(clf_loss,
#                                                          var_list=clf_vars):
#                 if self.debias:
#                     unit_adv_grad = normalize(adv_grads[var])
#                     # proj_{adv_grad} clf_grad:
#                     grad -= tf.reduce_sum(grad * unit_adv_grad) * unit_adv_grad
#                     if self.adversary_loss_weight is not None:
#                         grad -= self.adversary_loss_weight * adv_grads[var]
#                     else:
#                         grad -= tf.sqrt(global_step) * adv_grads[var]
#                 clf_grads.append((grad, var))

#             clf_min = clf_opt.apply_gradients(clf_grads,
#                                               global_step=global_step)
#             if self.debias:
#                 with tf.control_dependencies([clf_min]):
#                     adv_min = adv_opt.minimize(adv_loss, var_list=adv_vars)

#             self.sess_.run(tf.global_variables_initializer())

#             # Begin training
#             for epoch in range(self.num_epochs):
#                 shuffled_ids = rng.permutation(n_samples)
#                 for i in range(n_samples // self.batch_size):
#                     batch_ids = shuffled_ids[self.batch_size * i:
#                                              self.batch_size * (i+1)]
#                     batch_features = X[batch_ids]
#                     batch_labels = y[batch_ids][:, np.newaxis]
#                     batch_prot_attr = A[batch_ids][:, np.newaxis]
#                     batch_feed_dict = {self.input_ph: batch_features,
#                                        self.true_labels_ph: batch_labels,
#                                        self.prot_attr_ph: batch_prot_attr,
#                                        self.keep_prob: 0.8}
#                     if self.debias:
#                         _, _, clf_loss_val, adv_loss_val = self.sess_.run(
#                                 [clf_min, adv_min, clf_loss, adv_loss],
#                                 feed_dict=batch_feed_dict)

#                         if i % 200 == 0 and self.verbose:
#                             print("epoch {:>3d}; iter: {:>4d}; batch classifier"
#                                   " loss: {:.4f}; batch adversarial loss: "
#                                   "{:.4f}".format(epoch, i, clf_loss_val,
#                                                   adv_loss_val))
#                     else:
#                         _, clf_loss_val = self.sess_.run([clf_min, clf_loss],
#                                 feed_dict=batch_feed_dict)

#                         if i % 200 == 0 and self.verbose:
#                             print("epoch {:>3d}; iter: {:>4d}; batch classifier"
#                                   " loss: {:.4f}".format(epoch, i,
#                                                          clf_loss_val))

#         return self

#     def decision_function(self, X):
#         """Soft prediction scores.

#         Args:
#             X (pandas.DataFrame): Test samples.

#         Returns:
#             numpy.ndarray: Confidence scores per (sample, class) combination. In
#             the binary case, confidence score for ``self.classes_[1]`` where >0
#             means this class would be predicted.
#         """
#         check_is_fitted(self, ['classes_', 'input_ph', 'keep_prob',
#                                'classifier_logits_'])
#         n_samples = X.shape[0]
#         n_classes = len(self.classes_)
#         if n_classes == 2:
#             n_classes = 1

#         samples_covered = 0
#         scores = np.empty((n_samples, n_classes))
#         while samples_covered < n_samples:
#             start = samples_covered
#             end = samples_covered + self.batch_size
#             if end > n_samples:
#                 end = n_samples

#             batch_ids = np.arange(start, end)
#             batch_features = X.iloc[batch_ids]

#             batch_feed_dict = {self.input_ph: batch_features,
#                                self.keep_prob: 1.0}

#             scores[batch_ids] = self.sess_.run(self.classifier_logits_,
#                                                feed_dict=batch_feed_dict)
#             samples_covered += len(batch_features)

#         return scores.ravel() if scores.shape[1] == 1 else scores

#     def predict_proba(self, X):
#         """Probability estimates.

#         The returned estimates for all classes are ordered by the label of
#         classes.

#         Args:
#             X (pandas.DataFrame): Test samples.

#         Returns:
#             numpy.ndarray: Returns the probability of the sample for each class
#             in the model, where classes are ordered as they are in
#             ``self.classes_``.
#         """
#         decision = self.decision_function(X)

#         if decision.ndim == 1:
#             decision_2d = np.c_[np.zeros_like(decision), decision]
#         else:
#             decision_2d = decision
#         return scipy.special.softmax(decision_2d, axis=1)

#     def predict(self, X):
#         """Predict class labels for the given samples.

#         Args:
#             X (pandas.DataFrame): Test samples.

#         Returns:
#             numpy.ndarray: Predicted class label per sample.
#         """
#         scores = self.decision_function(X)
#         if scores.ndim == 1:
#             indices = (scores > 0).astype(int)
#         else:
#             indices = scores.argmax(axis=1)
#         return self.classes_[indices]