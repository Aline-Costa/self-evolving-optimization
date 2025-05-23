"""Validation"""
import time
import os
import logging
import argparse
import torch
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from mlp import MLP


# dataset definition
class CSVDataset(Dataset):
    """Data"""

    # load the dataset
    def __init__(self, path, device):
        # load the csv file as a dataframe
        data = np.genfromtxt(path, delimiter=",")
        # store the inputs, outputs ans masks
        self.x_train = data[:, :-2]
        self.y_train = data[:, -2]
        self.mask = data[:, -1]
        # ensure input data is floats
        self.x_train = torch.as_tensor(self.x_train).float().to(device)
        # label encode target and ensure the values are floats
        self.y_train = LabelEncoder().fit_transform(self.y_train)
        self.y_train = torch.as_tensor(self.y_train).to(device)
        self.shape = self.x_train.shape
        self.classes = torch.unique(self.y_train)
        self.n_classes = torch.numel(self.classes)
        self.chunk_size = 200

    # number of rows in the dataset
    def __len__(self):
        return len(self.x_train)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.x_train[idx], self.y_train[idx], self.mask[idx]]


def gmean_multiclass(preds_tensor, true_labels_tensor, classes_tensor):
    """prequential g-mean
    Returns a vector with gmean for each time step"""
    # Sensitivity of positive class = recall = true positive rate
    # Conventing to int and list to use dictionary keys
    classes = classes_tensor.int().tolist()
    preds = preds_tensor.int().tolist()
    true_labels = true_labels_tensor.int().tolist()
    preq_s_sens = {c: 0 for c in classes}
    preq_n_sens = {c: 0 for c in classes}
    running_sens = {c: 0 for c in classes}
    running_gmean = 0
    fading_factor = 0.999
    exp = 1 / len(classes)
    this_gmean_vector = []
    for y, f in zip(true_labels, preds):
        # Sensibility = true positive class rate
        test = int(f == y)  # test = true if correct prediction was made
        preq_s_sens[y] = test + fading_factor * preq_s_sens[y]
        preq_n_sens[y] = 1 + fading_factor * preq_n_sens[y]
        running_sens[y] = preq_s_sens[y] / preq_n_sens[y]
        # G-mean
        running_gmean = np.power(np.prod(list(running_sens.values())), exp)
        this_gmean_vector.append(running_gmean)
    return this_gmean_vector


def prequential_error_with_fading(predictions, true_labels, fading_factor=0.999):
    """Prequential error with fading factor for a batch of predictions"""
    preq_incorrect = 0
    preq_total = 0
    error_results = []

    for pred, true in zip(predictions, true_labels):
        preq_incorrect = fading_factor * preq_incorrect + int(pred != true)
        preq_total = fading_factor * preq_total + 1
        error = preq_incorrect / preq_total if preq_total != 0 else 0
        error_results.append(error)

    # Calculating accuracy from error
    accuracy_results = [1 - error for error in error_results]
    
    return accuracy_results


def chunks_define(dataset, ck_size):

    chunks = []
    for i in range(1,len(dataset.x_train)):
        
        if len(chunks) < ck_size: 
            chunk = dataset[0:i]
        else:    
            chunk = dataset[i-ck_size:i]
        chunks.append(chunk)
    return chunks


def evaluate_model(csv_file=None, config=None, epochs=30, device=None, stream_type=None, stream_name=None):  
    """Test Model"""
    dataset = CSVDataset(csv_file, device)
    train_ck = chunks_define(dataset, dataset.chunk_size)
    #train_dl = DataLoader(dataset, batch_size=200, shuffle=False)
    
    time_replicates = []

    #MLP
    gmean_replicates = []
    preds_replicates = []
    true_labels_replicates = []
    mean_gmean_replicates = []
    best_gmean = -1
    best_replicate = []
    best_model = None
    mean_accuracy_replicates = []
    prequential_accuracy_replicates = []

    #hidden1_size = 1
    hidden1_size = int(dataset.shape[1]/2)
    if dataset.shape[1] >= 6:
        hidden2_size = int(dataset.shape[1]/3)
    else:
        hidden2_size = hidden1_size

    #hidden_size = 25

    for epoch in range(epochs):
        logging.info("Starting epoch %s", epoch)

        # # define the network -> MLP
        mlp = MLP(dataset.shape[1], hidden1_size, hidden2_size, len(dataset.classes), device, **config)
        # train the model
        tic = time.process_time()
        true_labels, preds, mlp_ = mlp.train_evaluate(train_ck,mlp) 
        toc = time.process_time()
        elapsed_time = toc - tic

    # collect results

        #MLP

        # g-mean
        gmean_vector = gmean_multiclass(preds, true_labels, dataset.classes)
        gmean_replicates.append(gmean_vector)
        gmean = 100 * np.mean(gmean_vector)

        if gmean > best_gmean:
            best_gmean = gmean
            best_replicate = gmean_vector
            best_model= mlp_

        # prequential accuracy
        preq_accuracy = prequential_error_with_fading(preds, true_labels)

        mean_gmean_replicates.append(gmean)
        preds_replicates.append(preds.cpu().numpy())
        true_labels_replicates.append(true_labels.cpu().numpy())
        time_replicates.append(elapsed_time)

        mean_preq_accuracy = np.mean(preq_accuracy) if preq_accuracy else 0
        mean_accuracy_replicates.append(mean_preq_accuracy)
        prequential_accuracy_replicates.append(preq_accuracy)

    return (
        best_model,
        mlp.optimizer,
        {
            "mean_gmean": mean_gmean_replicates,
            "gmean": gmean_replicates,
            "preds": preds_replicates,
            "true_labels": true_labels_replicates,
            "best_replicate": best_replicate,
            "mean_preq_accuracy": mean_accuracy_replicates,
            "prequential_accuracy": prequential_accuracy_replicates
        }
    )
        

def run(stream_type,stream_name,path,config_path,result_path,models_path,epochs,device):
    """Run experiment for a particular stream"""
    # Reproducibility
    seed = int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Experiment params
    test_stream = os.path.join(path, f"data/{stream_type}/abrupt/abrupt_{stream_name}.csv")
    #test_stream = os.path.join(path, f"data/{stream_type}/{stream_name}.csv")  
    #test_stream = os.path.join(path, f"data/{stream_type}/streams_masks/{stream_name}.csv")
    best_config = np.load(
        os.path.join(config_path, f"config_agrawal_optim_adaptive.npz"),
        allow_pickle=True,
    )["best_config"].item()
    logging.info("Using best config: %s", best_config)
    
    best_model, optim, result = evaluate_model(csv_file=test_stream, config=best_config, epochs=epochs, 
                device=device, stream_type=stream_type, stream_name=stream_name)
    
    
    np.savez_compressed(
        os.path.join(result_path, f"mlp_{stream_type}_{stream_name}_{optim}"),
        best_config=best_config,
        seed=seed,
        **result,
    )
    torch.save(
        best_model.state_dict(),
        os.path.join(models_path, f"model_mlp_{stream_type}_{stream_name}_{optim}.pt"),
    )
    # Showing results
    logging.info("Finished MLP | data stream %s", stream_name)



def read_options(default_streams):
    """Read command line options"""
    path = os.path.abspath("../")
    result_path = os.path.join(path, "results/abrupt")
    config_path = os.path.join(path, "configs/abrupt")
    models_path = os.path.join(path, "models/abrupt")
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "-r",
        "--real",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        default=default_streams["real"],  # default if nothing is provided
    )
    cli.add_argument(
        "-a",
        "--artificial",
        nargs="*",
        default=default_streams["artificial"],
    )
    cli.add_argument(
        "-p",
        "--path",
        nargs="?",
        default=path,
    )
    cli.add_argument(
        "-c",
        "--config_path",
        nargs="?",
        default=config_path,
    )
    cli.add_argument(
        "-m",
        "--models_path",
        nargs="?",
        default=models_path,
    )
    cli.add_argument(
        "-o",
        "--result_path",
        nargs="?",
        default=result_path,
    )
    cli.add_argument(
        "-e",
        "--epochs",
        nargs="?",
        type=int,
        default=1,
    )
    args = cli.parse_args()
    return args


def main():
    """Calls experiments for all streams"""
    logging.basicConfig(
        encoding="utf-8",
        level=logging.INFO,
        format="[%(levelname)s %(asctime)s] %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S",
    )
    # result analysis setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # data streams
    streams = {
        "real": [        
            #"chess",
            #"keystroke",
            #"ozone8hr",
            #"luxembourg",
            #"elec2_uni_l5",
            #"noaa_uni_l5",
            #"powersupplyDayNight_uni_l5",
            #"sensorClasses29and31_uni_l5",
         
        ],
        "artificial": [
            #"sine1_uni_l5",
            #"sine2_uni_l5",
            #"agrawal1_uni_l5",
            #"agrawal2_uni_l5",
            #"agrawal3_uni_l5",
            "agrawal4_uni_l5",
            #"sea1_uni_l5",
            #"sea2_uni_l5",
            #"stagger1_uni_l5",
            #"stagger2_uni_l5",
        ],
    }
    args = read_options(streams)
    for stream_name in args.real:
        logging.info("Submitting MLP Real %s!", stream_name)
        run(
            "real",
            stream_name,
            args.path,
            args.config_path,
            args.result_path,
            args.models_path,
            args.epochs,
            device,
        )
    for stream_name in args.artificial:
        logging.info("Submitting MLP Artificial %s!", stream_name)
        run(
            "artificial",
            stream_name,
            args.path,
            args.config_path,
            args.result_path,
            args.models_path,
            args.epochs,
            device,
        )


if __name__ == "__main__":
    main()
