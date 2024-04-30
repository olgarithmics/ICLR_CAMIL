import pandas as pd
from args import parse_args
from models.camil import CAMIL
import os
from flushed_print import print
import tensorflow as tf
import numpy as np
import gc

def set_seed(seed: int = 42) -> None:
  import random
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DISABLE_SPARSE_SOFTMAX_XENT_WITH_LOGITS_OP_DETERMINISM_EXCEPTIONS']='1'

if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    adj_dim = None
    set_seed(12321)

    csv_file=args.csv_file
    acc=[]
    recall=[]
    f_score=[]
    auc=[]
    precision=[]

    fold_id = os.path.splitext(csv_file)[0].split("_")[3]

    references = pd.read_csv(csv_file)

    train_bags = references["train"].apply(lambda x: os.path.join(args.feature_path, x + ".h5")).values.tolist()

    def func_val(x):
        value = None
        if isinstance(x, str):
            value = os.path.join(args.feature_path, x + ".h5")
        return value

    val_bags = references.apply(lambda row: func_val(row.val), axis=1).dropna().values.tolist()

    test_bags = references.apply(lambda row: func_val(row.test), axis=1).dropna().values.tolist()

    train_net = CAMIL(args)

    train_net.train(train_bags, fold_id, val_bags, args)

    test_net = CAMIL(args)

    test_acc, test_auc = test_net.predict(test_bags,
                                                        fold_id,
                                                        args,
                                                        test_model=test_net.model)





