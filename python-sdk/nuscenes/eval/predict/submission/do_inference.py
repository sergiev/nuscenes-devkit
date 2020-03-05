# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
"""Script for generating a submission to the nuscenes prediction challenge."""
import argparse
import json
import os
from typing import List

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.predict.config import PredictionConfig
from nuscenes.eval.predict.config import load_prediction_config
from nuscenes.eval.predict.data_classes import Prediction
from nuscenes.eval.predict.splits import get_prediction_challenge_split
from nuscenes.predict import PredictHelper
from nuscenes.predict.input_representation.agents import \
    AgentBoxesWithFadedHistory
from nuscenes.predict.input_representation.combinators import Rasterizer
from nuscenes.predict.input_representation.interface import InputRepresentation
from nuscenes.predict.input_representation.static_layers import \
    StaticLayerRasterizer
from nuscenes.predict.models.backbone import ResNetBackbone
from nuscenes.predict.models.mtp import MTP, MTPLoss

NUM_MODES = 3


class MTPDataset(Dataset):
    """
    Implements a dataset for MTP.
    """

    def __init__(self, tokens: List[str], helper: PredictHelper,
                 input_representation: InputRepresentation):
        self.tokens = tokens
        self.helper = helper
        self.input_representation = input_representation

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item: int):

        instance_token, sample_token = self.tokens[item].split("_")

        img = self.input_representation.make_input_representation(instance_token, sample_token)
        img = torch.Tensor(img).permute(2, 0, 1)

        agent_state_vector = np.array([self.helper.get_velocity_for_agent(instance_token, sample_token),
                                           self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                           self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)])
        agent_state_vector = np.nan_to_num(agent_state_vector, 0.0)
        agent_state_vector = torch.Tensor(agent_state_vector)

        ground_truth = self.helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)

        return self.tokens[item], img, agent_state_vector, torch.FloatTensor(np.expand_dims(ground_truth, 0))


def load_model(helper: PredictHelper, config: PredictionConfig, path_to_model_weights: str):
    """
    Loads model with desired weights
    """
    backbone = ResNetBackbone('resnet50')
    model = nn.DataParallel(MTP(backbone, NUM_MODES))
    model.load_state_dict(torch.load(path_to_model_weights))
    return model


def do_inference_for_submission(helper: PredictHelper,
                                config: PredictionConfig,
                                path_to_model_weights: str,
                                dataset_tokens: List[str]) -> List[Prediction]:
    """
    Currently, this will make a submission with a constant velocity and heading model.
    Fill in all the code needed to run your model on the test set here. You do not need to worry
    about providing any of the parameters to this function since they are provided by the main function below.
    You can test if your script works by evaluating on the val set.
    :param helper: Instance of PredictHelper that wraps the nuScenes test set.
    :param config: Instance of PredictionConfig.
    :param path_to_model_weights: Path to model weights.
    :param dataset_tokens: Tokens of instance_sample pairs in the test set.
    :returns: List of predictions.
    """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtp_model = load_model(helper, config, path_to_model_weights)

    static_layer_rasterizer = StaticLayerRasterizer(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper)
    mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    val_dataset = MTPDataset(dataset_tokens, helper, mtp_input_representation)
    val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=16, drop_last=False, shuffle=False)

    mtp_model = mtp_model.eval()
    mtp_model = mtp_model.to(device)

    prediction_list = []

    for index, data in tqdm.tqdm(enumerate(val_dataloader)):
        tokens, img, agent_state_vector, ground_truth = data

        img = img.to(device)
        agent_state_vector = agent_state_vector.to(device)
        ground_truth = ground_truth.to(device)

        prediction = mtp_model(img, agent_state_vector)

        prediction = prediction.cpu().detach().numpy()
        
        mode_probs = prediction[:, -NUM_MODES:]
        trajectories = prediction[:, :-NUM_MODES].reshape(-1, 3, 12, 2)
        
        for i, token in enumerate(tokens):
            instance, sample = token.split("_")

            pred = Prediction(instance, sample, trajectories[i], mode_probs[i])

            prediction_list.append(pred)

    return prediction_list


def main(version: str, data_root: str, split_name: str, model_weights: str,
         output_dir: str, submission_name: str, config_name: str) \
        -> None:
    """
    Makes predictions for a submission to the nuScenes prediction challenge.
    :param version: NuScenes version.
    :param data_root: Directory storing NuScenes data.
    :param split_name: Data split to run inference on.
    :param model_weights: Path to model weights.
    :param output_dir: Directory to store the output file.
    :param submission_name: Name of the submission to use for the results file.
    :param config_name: Name of config file to use.
    """
    nusc = NuScenes(version=version, dataroot=data_root)
    helper = PredictHelper(nusc)
    dataset = get_prediction_challenge_split(split_name)
    config = load_prediction_config(helper, config_name)

    predictions = do_inference_for_submission(helper, config, model_weights, dataset)
    predictions = [prediction.serialize() for prediction in predictions]
    json.dump(predictions, open(os.path.join(output_dir, f"{submission_name}_inference.json"), "w"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Perform Inference for submission.')
    parser.add_argument('--version', help='NuScenes version number.')
    parser.add_argument('--data_root', help='Root directory for NuScenes json files.')
    parser.add_argument('--split_name', help='Data split to run inference on.')
    parser.add_argument('--model_weights', help='Path to model weights')
    parser.add_argument('--output_dir', help='Directory to store output file.')
    parser.add_argument('--submission_name', help='Name of the submission to use for the results file.')
    parser.add_argument('--config_name', help='Name of the config file to use', default='predict_2020_icra.json')

    args = parser.parse_args()
    main(args.version, args.data_root, args.split_name, args.model_weights,
         args.output_dir, args.submission_name, args.config_name)
