"""
Learning and inference on NSL-KDD
=================================

The NSL-KDD dataset is an improvement over the KDD'99 provided by the Canadian
Institute for Cybersecurity.


From their `webpage <https://www.unb.ca/cic/datasets/nsl.html>`__:

    *NSL-KDD is a data set suggested to solve some of
    the inherent problems of the KDD'99 data set which are mentioned in [1].
    Although, this new version of the KDD data set still suffers from some
    of the problems discussed by McHugh and may not be a perfect
    representative of existing real networks, because of the lack of public
    data sets for network-based IDSs, we believe it still can be applied as
    an effective benchmark data set to help researchers compare different
    intrusion detection methods.*

    *Furthermore, the number of records in the
    NSL-KDD train and test sets are reasonable. This advantage makes it
    affordable to run the experiments on the complete set without the need
    to randomly select a small portion. Consequently, evaluation results of
    different research work will be consistent and comparable.*

    *[1] M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, "A Detailed Analysis
    of the KDD CUP 99 Data Set," Submitted to Second IEEE Symposium on
    Computational Intelligence for Security and Defense Applications
    (CISDA), 2009.*

This tutorial is a demonstration of how to use the Akida Execution
Engine to classify connexions into 5 classes: normal activity and 4
different groups of attacks (DoS, U2R, R2L and probe).

"""

######################################################################
# 1. Download and prepare the NSL-KDD dataset
# -------------------------------------------

# Various imports needed for the tutorial
import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_file
from sklearn import preprocessing
from sklearn.utils import shuffle

# Akida specific imports
from akida import Model, InputData, FullyConnected, LearningType, dense_to_sparse

# Cybersecurity specific imports
from progressbar import ProgressBar
from sklearn.metrics import confusion_matrix, f1_score, precision_score, \
    recall_score, accuracy_score

######################################################################

# Retrieve NSLKDD dataset
file_path = get_file('NSL-KDD.zip', 'https://iscxdownloads.cs.unb.ca/iscxdownloads/NSL-KDD/NSL-KDD.zip', cache_subdir='datasets/NSL-KDD', extract=True)
working_dir = os.path.dirname(file_path)
print ('Using NSL-KDD dataset')

######################################################################

# Define data set column names
col_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"])

######################################################################

# Funtion used to prepare the NSL-KDD dataset
def data_prep_NSLKDD(data):
    """
    Rename group original attack names in 4 categories: DoS, Probe, R2L, U2R
    Perform LabelEncoding() on the label column (convert strings to numbers)
    Remove features/columns with a cardinality of 1 (= all cells are identical)

    Args:
        data: NSLKDD dataset
    """
    # Dictionary that contains mapping of various attacks to the four main
    # categories
    attack_dict = {
        'normal': 'normal',

        'back': 'DoS',
        'land': 'DoS',
        'neptune': 'DoS',
        'pod': 'DoS',
        'smurf': 'DoS',
        'teardrop': 'DoS',
        'mailbomb': 'DoS',
        'apache2': 'DoS',
        'processtable': 'DoS',
        'udpstorm': 'DoS',

        'ipsweep': 'Probe',
        'nmap': 'Probe',
        'portsweep': 'Probe',
        'satan': 'Probe',
        'mscan': 'Probe',
        'saint': 'Probe',

        'ftp_write': 'R2L',
        'guess_passwd': 'R2L',
        'imap': 'R2L',
        'multihop': 'R2L',
        'phf': 'R2L',
        'spy': 'R2L',
        'warezclient': 'R2L',
        'warezmaster': 'R2L',
        'sendmail': 'R2L',
        'named': 'R2L',
        'snmpgetattack': 'R2L',
        'snmpguess': 'R2L',
        'xlock': 'R2L',
        'xsnoop': 'R2L',
        'worm': 'R2L',

        'buffer_overflow': 'U2R',
        'loadmodule': 'U2R',
        'perl': 'U2R',
        'rootkit': 'U2R',
        'httptunnel': 'U2R',
        'ps': 'U2R',
        'sqlattack': 'U2R',
        'xterm': 'U2R'
    }
    data["label"].replace(attack_dict, inplace=True)

    # Label encoding
    le = preprocessing.LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    Label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

    # Drop columns which are all zeros in the training set
    cardinalities = data.apply(pd.Series.nunique)
    good_columns = cardinalities>1
    data = data.loc[:, good_columns]

    return(data, Label_mapping)

######################################################################

# Load the dataset
train_data = pd.read_csv(os.path.join(working_dir, 'KDDTrain+.txt'), sep=',', header=None, names=col_names, index_col=False)
train_data['split'] = 'train'
test_data = pd.read_csv(os.path.join(working_dir, 'KDDTest+.txt'), sep=',', header=None, names=col_names, index_col=False)
test_data['split'] = 'test'

# Prepare the dataset
data = pd.concat([train_data, test_data])
data, Label_mapping = data_prep_NSLKDD(data)

# Resplit train and test sets
y_train_df = data.label[data.split=='train']
X_train_df = data[data.split=='train'].drop(['label','split'],axis=1)

y_test_df = data.label[data.split=='test']
X_test_df = data[data.split=='test'].drop(['label','split'],axis=1)


######################################################################
# 2. Sneak peek of the input tabular data
# ---------------------------------------

# Display train set shape and data sneak peek
print ('Train set shape: %s' % (X_train_df.shape,))
print(X_train_df.head())


######################################################################
# 3. Convert from tabular to binary data
# --------------------------------------

######################################################################
# The ``onehotencode_df()`` function will transform these
# tabular data to "binary" data using a one-hot encoding scheme for all
# variables. This technique is usually the way to go only for categorical
# variables. Here, since we want to learn from binary data, we will use
# this technique also for the other variable types in ``X_train``:
#
#   * For continuous numeric variables, we will use binning + one-hot-encoding.
#     This process is akin to a Gaussian Receptive Field *data to spike*
#     conversion in our case. The only difference between the classical GRF
#     and the current implementation is that we did not implement overlap
#     between bins. This may cancel the *encoding precision for free* you can
#     get with overlap. But on the other hand this allows to know exactly the
#     number of positive activations. Thus, the balance is probably positive
#     here.
#   * For boolean variables, we used 2 neurons for each variable. Only
#     one may have been used. But having two again allows to know exactly the
#     number of positive activations.
#
# As already said, one nice feat of this scheme is that the number of
# positive activations is fixed: it is exactly one per original variable.
# This is convenient for Akida native learning.

def compute_df_mappings(df, n_bins=20, method='quantized'):
    """ Compute mappings to tranform columns of a Pandas dataframe to be Akida-ready.

    The mappings type depends on the column dtype:
    - numerical: bin edges for bucketizing
    - categorical: list of modalities for one-hot encoding
    - boolean: same as categorical

    :param df: raw tabular data, standard X in machine learning for
        structured data. Should be a Pandas Dataframe (which is
        standard now): one line per instance, one column per feature.
    :param n_bins: number of bins used for numerical columns.
        Default is 20.
    :param method: method used to make the binning. If 'quantized' then
        it makes the bin probability uniform on the train set. If 'hist'
        it keep the original distribution intact.
        Default is 'quantized'

    :return: A tuple containing the mappings, the recognized type
        of each column, and the new column names for the "extended" X.
    """

    # Define quantiles
    myquantiles = np.linspace(0,1,n_bins+1)

    mappings = dict()
    used_type = dict()
    n_col = len(df.columns)

    col_names = []
    for col in df.columns:
        kind = df[col].dtype.kind
        cardinality = df[col].nunique()
        print("Compute mappings for column *{}* of kind {} with cardinality {}".format(col,kind,cardinality))
        if cardinality<=n_bins: kind = 'O' # do as if it was an object for low cardinality col
        if kind == 'O':
            X, mapping_index = pd.Series(df[col]).factorize()
            mappings[col] = dict(zip(mapping_index, range(len(mapping_index))))
            col_names = col_names + [str(col) + '_' + str(s) for s in mappings[col]]
            used_type[col] = 'cat'
        elif kind == 'b':
            mappings[col] = {False:0, True:1} # always the same, no need to compute
            col_names = col_names + [str(col) + '_' + s for s in ['0','1']]
            used_type[col] = 'boo'
        else: # if continuous/numerical
            if method=='quantized':
                bin_edges = np.nanquantile(df[col], myquantiles) # make the distribution uniform
                bin_edges = np.delete(bin_edges,[0,len(myquantiles)-1])
            elif method=='hist':
                _,bin_edges = np.histogram(df[col], bins=n_bins) # keep distribution as it is
                bin_edges = np.delete(bin_edges,[0,len(myquantiles)-1])
            mappings[col] = np.unique(bin_edges) # Merge identical bin_edges
            col_names = col_names + [str(col) + '_bin_' + str(x) for x in range(len(mappings[col])+1)]
            used_type[col] = 'num'
    print("Done.", end='\n')
    print("Information for Akida yml configuration: inputWidth = {} and numWeights = {}.".format(len(col_names), len(df.columns)))
    return mappings, used_type, col_names

######################################################################

def onehotencode_df(df, mappings=None, n_bins=20, verbose=0):
    """ Convert tabular/structured data (=dataframe) to spike for Akida.

    This function transforms a Pandas dataframe to be Akida-ready using
    one-hot encoding on virtually all kind of input features (numerical,
    boolean, categorical).

    For numerical features, since it bucketizes the data based on quantiles,
    it thus also "normalizes" the data distribution: there is no need for
    scaling or procedure to take care of outliers).

    The input data should be NaN-free.

    NOTES:
    - This function may be rewritten to work as a layer.
    - No overlap in binning (different from GRF). Is it a problem? Apparently not.

    :param df: raw tabular data, standard X in machine learning for
        structured data. Should be a Pandas Dataframe (which is standard
        now): one line per instance, one column per feature.
    :param mappings: a tuple with the parameters of the transformations
        for each column. See compute_df_mappings() on how to build it,
        defaults to None (compute_df_mappings() is then performed first
        here).
    :param n_bins: number of bins used for numerical columns. Default is 20.
        Only used if mappings==None.
    :param verbose: set verbosity. Default is 0 (= no output).

    :return: A tuple containing the binarized df (as a boolean numpy array,
        same number of than the input df) and a tuple with the parameters of
        the transformations for each column.
    """

    if mappings is None:
        mappings = compute_df_mappings(df, n_bins=20)

    X = np.array([], dtype=np.bool_).reshape(len(df),0) # init w/ good length
    if verbose>0: print("Convert column: ", end="")
    for col in df.columns:
        if verbose>0: print("{}".format(col), end=', ')
        if mappings[1][col] == 'cat':
            X = np.hstack((X, onehotencode_categorical_column(df[col], mapping_index=mappings[0][col])))
        elif mappings[1][col] == 'boo':
            X = np.hstack((X, onehotencode_boolean_column(df[col])))
        elif mappings[1][col] == 'num':
            X = np.hstack((X, onehotencode_numerical_column(df[col], bin_edges=mappings[0][col])))
        else:
            print('Warning: unknown column format. Try rerunning compute_df_mappings().')
    if verbose>0: print("Done. X is now {}".format(X.shape))
    return X, mappings


def onehotencode_numerical_column(X, bin_edges):
    """
    Transform a numerical vector to Akida-ready data using one-hot encoding

    :param X: A vector of values (typically a Pandas column/series from a dataframe).
    :param bin_edges: precomputed bin edges for the bucketizing.

    :return: A boolean numpy matrix corresponding to a one-hot encoded input X.
    """
    X = np.digitize(X, bin_edges) # ~label encoding
    X = np.eye(len(bin_edges)+1)[X] # make it one-hot
    return X.astype(np.bool_)

def onehotencode_boolean_column(X):
    """
    Transform a boolean-like vector to Akida-ready data using one-hot encoding

    :param X: A vector of values that are boolean-like (= contains only
        zeros and ones, they do not need to have a dtype==bool_).

    :return: A boolean numpy matrix (same number of lines than input X, 2 columns)
        corresponding to a one-hot encoded input X.
    """
    X = np.eye(2)[np.asarray(X).astype(np.uint8)]
    return X.astype(np.bool_)

def onehotencode_categorical_column(X, mapping_index):
    """
    Transform a categorical vector to Akida-ready data using one-hot encoding

    :param X: A vector of values (typically a Pandas column/series from a dataframe).
    :param mapping_index: precomputed bin edges for the bucketizing.

    :return: A boolean numpy matrix corresponding to a one-hot encoded input X.
    """
    remove_last_column = False
    X.iloc[:].loc[~X.isin(mapping_index.keys())] = -1 # values not in mapping are replaced with -1
    X = X.replace(mapping_index)
    n_col = len(mapping_index)
    if np.any(X==-1):
        remove_last_column = True
        n_col = n_col+1

    X = np.eye(n_col)[X.astype(int)]
    if remove_last_column:
        X = np.delete(X, -1, axis=1) # remove the -1 column (=values not seen during training)

    return X.astype(np.bool_)

######################################################################

# Convert from tabular to binary data
mappings = compute_df_mappings(X_train_df, n_bins=20)
X_train, mappings = onehotencode_df(X_train_df, mappings=mappings)
X_test, mappings = onehotencode_df(X_test_df, mappings=mappings)

######################################################################
# .. Note:: The output printed from the cell above should be used to define
#           the architecture and the parameters of the Akida model.

######################################################################
# 4. Oversampling the training data to cope with imbalanced dataset
# -----------------------------------------------------------------

######################################################################
# Since there is much more exemplars of normal activity than any other
# attacks, we used over-sampling. This technique consists in duplicating
# exemplars of the classes less represented in the training set so that
# all classes are equally represented.

# Re-sampling specific import
from imblearn.over_sampling import RandomOverSampler

######################################################################

# Check original distribution
print('Classes and their frequencies in the dataset:')
print(y_train_df.value_counts(normalize=True, sort=False))

# Convert label series to numpy arrays
y_train = y_train_df.to_numpy()
y_test = y_test_df.to_numpy()

# Oversampling
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)
X_train, y_train = shuffle(X_train, y_train)

print('Classes and their frequencies in the dataset (after oversampling):')
print(pd.Series(y_train).value_counts(normalize=True, sort=False))


######################################################################
# 5. Configuring Akida model
# --------------------------

######################################################################
# To modify Akida architecture and parameters, edit model below.

#Create a model
model = Model()
model.add(InputData("input", input_width=312, input_height=1, input_features=1))
fully = FullyConnected("fully", num_neurons=10240, activations_enabled=False)
model.add(fully)
# Configure the last layer for semi-supervised training
fully.compile(num_weights=40, num_classes=5)

model.summary()


######################################################################
# 6. Learning and inference
# -------------------------

######################################################################
# Depending on your setup, training the Akida model will take some time

def convert_dataset_to_spikes(X):
    X_spikes = []
    for i in ProgressBar()(range(X.shape[0])):
        sample = X[i].reshape((X[i].shape[0], 1, 1))
        X_spikes.append(dense_to_sparse(sample))
    return X_spikes

print("Convert the train set to spikes")
X_train_spikes = convert_dataset_to_spikes(X_train)

print("Perform training one sample at a time")
for i in ProgressBar()(range(len(X_train_spikes))):
    model.fit(X_train_spikes[i], input_labels=y_train[i])

######################################################################
# 7. Display results
# ------------------

######################################################################

# Define a method to compute performances
def CS_performance_measures(y_true, y_pred, labels=None, normal_class=0):
    """ Cybersecurity / anomaly detection custom performance measures.

    These specific performance measures are designed to assess precisely
    the performance in the case of a multiclass classification task in
    which one class is "special". For cybersecurity for example the 'normal'
    activity is common and a special class in comparison to different
    type of attacks. This class is here indexed as the normal_class variable.

    What happens here is that performance measures are computed at two nested
    levels:
    1) All attacks are merged together and measures are computed
    (i.e. 0 vs. [1,2,3,4] as 1)
    2) Measured are computed on attacks only (i.e. 1 vs. 2 vs. 3 vs. 4)

    Relies on sklearn.metrics.

    labels is a list of labels ids. If labels and their original names are saved in
    a dictionnary with the class names as keys and the ids as index
    (e.g. labels = {'DoS': 0, 'Probe': 1, 'R2L': 2, 'U2R': 3, 'normal': 4}),
    then labels = list(labels.values()).

    normal_class is the index of the class considered as normal (vs. e.g. attack or
    failure depending on the use case)

    Note that in binary classification, recall of the positive class is also
    known as sensitivity; recall of the negative class is specificity.
    """

    # Make the new (binary classif) vectors corresponding to attack vs normal
    attack_true = np.full(len(y_true), np.nan)
    attack_true[y_true == normal_class] = 0
    attack_true[y_true != normal_class] = 1
    attack_pred = np.full(len(y_pred), np.nan)
    attack_pred[y_pred == normal_class] = 0
    attack_pred[y_pred != normal_class] = 1

    normal_lines = np.logical_or(y_true==0, y_pred==0)
    attack_lines = ~normal_lines

    results = {"accuracy": accuracy_score(attack_true, attack_pred),
               "f1": f1_score(attack_true, attack_pred, average='weighted'),
               "recall-sensitivity": recall_score(attack_true, attack_pred, pos_label=1),
               "recall-specificity": recall_score(attack_true, attack_pred, pos_label=0),
               "precision": precision_score(attack_true, attack_pred, average='weighted'),

               "accuracy_among_attacks": accuracy_score(y_true[attack_lines], y_pred[attack_lines]),
               "f1_among_attacks": f1_score(y_true[attack_lines], y_pred[attack_lines], average='weighted'),
               "recall_among_attacks": recall_score(y_true[attack_lines], y_pred[attack_lines], average='weighted'),
               "precision_among_attacks": precision_score(y_true[attack_lines], y_pred[attack_lines], average='weighted'),

               "cm": confusion_matrix(y_true, y_pred, labels)}
    return results

######################################################################

# Check performances against the test set
res = pd.DataFrame()

print("Convert the test set to spikes")
X_test_spikes = convert_dataset_to_spikes(X_test)

print("Classify test samples")
y_pred = np.empty((0, 1), dtype=int)
for i in ProgressBar()(range(len(X_test_spikes))):
    outputs = model.predict(X_test_spikes[i], 5)
    y_pred = np.append(y_pred, outputs)
results = CS_performance_measures(y_test, y_pred, list(Label_mapping.values()), normal_class=4)

# Display results
res = res.append(results, ignore_index=True)
print("Accuracy: "+"{0:.2f}".format(100*results["accuracy"])+"% / "+"F1 score: "+"{0:.2f}".format(results["f1"]))

# For non-regression purpose
assert results["accuracy"] > 0.85

# Get model statistics on a few samples
stats = model.get_statistics()
for i in range(20):
    model.forward(X_test_spikes[i])

# Print model statistics
print("Model statistics")
for _, stat in stats.items():
    print(stat)

######################################################################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Taken from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.autoscale()

######################################################################

# Display normalized confusion matrix
plt.rcParams['figure.figsize'] = [8, 4]
plt.figure()
plot_confusion_matrix(res['cm'][len(res)-1], classes=Label_mapping, normalize=True,
                      title='Normalized confusion matrix')

######################################################################

# Display confusion matrix, raw numbers
plt.figure()
plot_confusion_matrix(res['cm'][len(res)-1], classes=Label_mapping,
                      title='Confusion matrix, raw numbers')
