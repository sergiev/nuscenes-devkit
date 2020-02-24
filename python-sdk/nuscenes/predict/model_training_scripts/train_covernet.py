# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

"""
Regression test to see if MTP can overfit on a single example.
"""

import argparse
import os
import pickle
from typing import List
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch import nn

from nuscenes import NuScenes
from nuscenes.predict import PredictHelper
from nuscenes.eval.predict.splits import get_prediction_challenge_split
from nuscenes.predict.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.predict.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.predict.input_representation.interface import InputRepresentation
from nuscenes.predict.input_representation.combinators import Rasterizer
from nuscenes.predict.models.backbone import ResNetBackbone
from nuscenes.predict.models.covernet import CoverNet, ConstantLatticeLoss


class CoverNetDataset(Dataset):
    """
    Implements a dataset for CoverNet.
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
        agent_state_vector = np.nan_to_num(agent_state_vector, -10.0)
        agent_state_vector = torch.Tensor(agent_state_vector)

        ground_truth = self.helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)

        return img, agent_state_vector, torch.FloatTensor(np.expand_dims(ground_truth, 0))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train MTP.')
    parser.add_argument('--num_epochs', type=int, help='Number of Epochs to train for')
    parser.add_argument('--nuscenes_version', default='v1.0-trainval')
    parser.add_argument('--nuscenes_dataroot')
    parser.add_argument('--split_name', default='', choices=['mini', ''])
    parser.add_argument('--loss_file_name', help='File to store the loss after every epoch.')
    parser.add_argument('--lattice_pickle_file', help='Pickle file storing the lattice.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    loss_file_name = os.path.join(args.output_dir, args.loss_file_name)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    nusc = NuScenes(args.nuscenes_version, dataroot=args.nuscenes_dataroot)
    helper = PredictHelper(nusc)

    if args.split_name == 'mini':
        prefix = 'mini_'
    else:
        prefix = ''

    def filter_tokens(tokens, helper: PredictHelper):
        return [tok for tok in tokens if 'vehicle' in helper.get_sample_annotation(*tok.split("_"))['category_name']]

    train_tokens = filter_tokens(get_prediction_challenge_split(prefix + 'train'), helper)
    val_tokens = filter_tokens(get_prediction_challenge_split(prefix + 'val'), helper)

    static_layer_rasterizer = StaticLayerRasterizer(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper)
    covernet_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    train_dataset = CoverNetDataset(train_tokens, helper, covernet_input_representation)
    val_dataset = CoverNetDataset(val_tokens, helper, covernet_input_representation)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    lattice = pickle.load(open(args.lattice_pickle_file, "rb"))

    backbone = ResNetBackbone('resnet50')
    model = nn.DataParallel(CoverNet(backbone, lattice.shape[0]))
    model = model.to(device)

    loss_function = ConstantLatticeLoss(lattice)

    losses = {'train': [],
              'val': []}

    json.dump(losses, open(loss_file_name, "w"))

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch_number in range(args.num_epochs):

        train_loss = 0
        val_loss = 0
        for index, data in enumerate(train_dataloader):
            img, agent_state_vector, ground_truth = data

            img = img.to(device)
            agent_state_vector = agent_state_vector.to(device)
            ground_truth = ground_truth.to(device)

            optimizer.zero_grad()

            prediction = model(img, agent_state_vector)

            loss = loss_function(prediction, ground_truth)
            loss.backward()
            optimizer.step()

            current_loss = loss.cpu().detach().numpy()

            print(f"Current train loss at epoch {epoch_number} and iteration {index} is {current_loss:.4f}")
            train_loss += current_loss

        epoch_train_loss = train_loss / (index + 1)

        losses = json.load(open(loss_file_name))
        losses['train'].append(epoch_train_loss)
        json.dump(losses, open(loss_file_name, "w"))

        torch.save(model.state_dict(), os.path.join(args.output_dir, f'./epoch{epoch_number}.pth'))

        with torch.no_grad():
            for index, data in enumerate(val_dataloader):
                img, agent_state_vector, ground_truth = data
                img = img.to(device)
                agent_state_vector = agent_state_vector.to(device)
                ground_truth = ground_truth.to(device)

                prediction = model(img, agent_state_vector)
                loss = loss_function(prediction, ground_truth)
                current_loss = loss.cpu().detach().numpy()

                print(f"Current val loss at epoch {epoch_number} and iteration {index} is {current_loss:.4f}")
                val_loss += current_loss

        epoch_val_loss = val_loss / (index + 1)

        losses = json.load(open(loss_file_name))
        losses['val'].append(epoch_val_loss)
        json.dump(losses, open(loss_file_name, "w"))






