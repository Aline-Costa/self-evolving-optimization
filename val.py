"""Validation with hyperopt"""
import os
import time
from functools import partial
import logging
import torch
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, space_eval
from run import evaluate_model



def evaluate(config, data_file=None, epochs=1, device=None):
    """Function to be minimized"""
    config = {**config, **config["optimizer_type"]}
    config.pop("optimizer_type", None)
    print(config)
    
    _, optim,result = evaluate_model(
        csv_file=data_file,
        config=config,
        epochs=epochs,
        device=device,
    )
    acc = result["mean_gmean"][0]
    return {"loss": -acc, "status": STATUS_OK}
    

def model_selection(data_file=None, epochs=1, max_evals=500, device=None):
    """Model selection"""
    config = {
        "optimizer_type": hp.choice(
            "optimizer_type",
            [
               {
                    "optimizer": "sgd",
                    "lr": hp.uniform("sgd_lr", 1e-5, 1),
                    "momentum": hp.uniform("momentum", 0, 1),
                    "nesterovs_momentum": hp.choice(
                        "nesterovs_momentum", [False, True]),
                    "weight_decay": hp.uniform("sgd_weight_decay", 0, 1),
               },
               #{
               #     "optimizer": "adam",
               #     "beta_1": hp.uniform("adam_beta_1", 0, 1),
               #     "beta_2": hp.uniform("adam_beta_2", 0, 1),
               #     "lr": hp.uniform("adam_lr", 1e-5, 1),
               #     "weight_decay": hp.uniform("adam_weight_decay", 0, 1),
               #},
                #{
                #   "optimizer": "adamw",
                #     "beta_1": hp.uniform("adamw_beta_1", 0, 1),
                #     "beta_2": hp.uniform("adamw_beta_2", 0, 1),
                #     "lr": hp.uniform("adamw_lr", 1e-5, 1),
                #    "weight_decay": hp.uniform("adamw_weight_decay", 0, 1),
                #},
                #{
                #     "optimizer": "adadelta",
                #    "weight_decay": hp.uniform("adadelta_weight_decay", 0, 1),
                #},
                
                #{
                #     "optimizer": "rmsprop",
                #     "momentum": hp.uniform("rmsprop_momentum", 0, 1),
                #     "alpha": hp.uniform("rmsprop_alpha", 0, 1),
                #     "lr": hp.uniform("rmsprop_lr", 1e-2, 1),
                #     "weight_decay": hp.uniform("rmsprop_weight_decay", 0, 1),
                #},
                #{
                #    "optimizer": "optim_adaptive",
                #    "beta1": hp.uniform("optim_beta_1", 0, 1),
                #    "beta2": hp.uniform("optim_beta_2", 0, 1),
                #    "lr": hp.uniform("optim_lr", 1e-5, 1),
                #    "weight_decay": hp.uniform("optim_weight_decay", 0, 1),
                #},
                
            ],
        ),
    }
    func_eval = partial(
        evaluate,
        data_file=data_file,
        epochs=epochs,
        device=device,
    )
    best_config = fmin(
        fn=func_eval, space=config, algo=tpe.suggest, max_evals=max_evals
    )
    # saving
    logging.info("FOUND BEST CONFIG %s", best_config)
    return space_eval(config, best_config)


def main():
    """Main function"""
    logging.basicConfig(
        encoding="utf-8",
        # level=logging.INFO,
        format="[%(levelname)s %(asctime)s] %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S",
    )
    # Reproducibility
    seed = int(time.time())
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Experiment params
    max_evals = 500
    # result analysis setup
    path = os.path.abspath("../")
    config_path = os.path.join(path, "configs/")
    # datasets
    streams = streams = {
        "real": [
            #"elec2",
            #"noaa",
            #"powersupplyDayNight",
            #"sensorClasses29and31",
            #"keystroke",
            #"chess",
            #"ozone8hr",
            #"luxembourg",
        ],
        "artificial": [
            #"agrawal",
            #"sea",
            #"sine",
            #"stagger",
        ],
    }
    for data_type, names in streams.items():
        for stream_name in names:
            val_stream = os.path.join(path, f"data/{data_type}/val/val_{stream_name}.csv")
            print(val_stream)
            print(f"VALIDATING MLP ON {data_type} {stream_name}!!")
            best_config = model_selection(
                data_file=val_stream,
                epochs=1,
                max_evals=max_evals,
                device=device,
            )
            best_config = {
                **best_config,
                **best_config["optimizer_type"],
            }
            best_config.pop("optimizer_type", None)
            np.savez_compressed(
                os.path.join(config_path, f"config_{stream_name}_sgd"),
                best_config=best_config,
                seed=seed,
            )


if __name__ == "__main__":
    main()
