import argparse

def parse_args():
    """Parse input arguments.
    Parameters
      -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class objecttransmil_ac
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Train CAMIL')
    parser.add_argument('--save_dir', dest='save_dir',help='directory where the weights of the model are stored',default="saved_models", type=str)
    parser.add_argument('--lr', dest='init_lr',help='initial learning rate',default=0.0002, type=float)
    parser.add_argument('--decay', dest='weight_decay',help='weight decay',default=1e-5, type=float)
    parser.add_argument('--thresh',  help='thresh', default=0.5, type=float)
    parser.add_argument('--epochs', dest='epochs',help='number of epochs to train CAMIL',default=30, type=int)
    parser.add_argument('--seed_value', dest='seed_value',help='use same seed value for reproducability',default=12321, type=int)
    parser.add_argument('--feature_path', dest='feature_path',help='directory where the images are stored',default="h5_files", type=str)
    parser.add_argument('--dataset_path', dest='data_path',help='directory where the images are stored',default="slides",type=str)
    parser.add_argument('--experiment_name', dest='experiment_name',help='the name of the experiment needed for the logs', default="test", type=str)
    parser.add_argument('--input_shape', dest="input_shape",help='shape of the image',default=(512,), type=int, nargs=3)
    parser.add_argument('--label_file', dest="label_file",help='csv file with information about the labels',default="label_files/camelyon_data.csv",type=str)
    parser.add_argument('--csv_file', dest="csv_file", help='csv file with information about the labels',default="camelyon_csv_splits/splits_0.csv",type=str)
    parser.add_argument('--raw_save_dir', dest="raw_save_dir", help='directory where the attention weights are saved', default="heatmaps", type=str)
    parser.add_argument('--retrain', dest="retrain", action='store_true', default=False)
    parser.add_argument('--save_exp_code', type=str, default=None,help='experiment code')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
    parser.add_argument('--subtyping', dest="subtyping",action='store_true', default=False)
    parser.add_argument('--n_classes', default=2, type=int)


    args = parser.parse_args()
    return args

