# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

"""
Regression test to see if MTP can overfit on a single example.
"""

import argparse
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
from nuscenes.predict.models.mtp import MTP, MTPLoss


class MTPDataset(Dataset):
    """
    Implements a dataset for MTP.
    """

    def __init__(self, tokens: List[str], helper: PredictHelper,
                 input_representation: InputRepresentation,
                 num_modes: int = 1):
        self.tokens = tokens
        self.helper = helper
        self.input_representation = input_representation
        self.num_modes = num_modes

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item: int):

        instance_token, sample_token = self.tokens[item].split("_")

        img = self.input_representation.make_input_representation(instance_token, sample_token)
        img = torch.Tensor(img).permute(2, 0, 1)

        agent_state_vector = torch.Tensor([self.helper.get_velocity_for_agent(instance_token, sample_token),
                                           self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                           self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)])

        ground_truth = self.helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)

        return img, agent_state_vector, torch.FloatTensor(np.expand_dims(ground_truth, 0))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train MTP.')
    parser.add_argument('--num_epochs', type=int, help='Number of Epochs to train for')
    parser.add_argument('--nuscenes_version', default='v1.0-trainval')
    parser.add_argument('--split_name', default='')
    parser.add_argument('--loss_file_name', help='File to store the loss after every epoch.')
    parser.add_argument('--num_modes', type=int, help='How many modes to learn.', default=1)
    parser.add_argument('--use_gpu', type=bool, help='Whether to use gpu', default=False)
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    nusc = NuScenes(args.nuscenes_version)
    helper = PredictHelper(nusc)

    if args.split_name == 'mini':
        prefix = 'mini_'
    else:
        prefix = ''

    def filter_tokens(tokens, helper: PredictHelper):
        return [tok for tok in tokens if 'vehicle' in helper.get_sample_annotation(*tok.split("_"))['category_name']][:48]
    

    train_tokens = filter_tokens(get_prediction_challenge_split(prefix + 'train'), helper)
    val_tokens = filter_tokens(get_prediction_challenge_split(prefix + 'val'), helper)

    static_layer_rasterizer = StaticLayerRasterizer(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper)
    mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    train_dataset = MTPDataset(train_tokens, helper, mtp_input_representation, args.num_modes)
    val_dataset = MTPDataset(val_tokens, helper, mtp_input_representation, args.num_modes)
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=16)

    backbone = ResNetBackbone('resnet50')
    model = nn.DataParallel(MTP(backbone, args.num_modes))
    model = model.to(device)

    loss_function = MTPLoss(args.num_modes, 1, 5)

    losses = {'train': [],
              'val': []}

    json.dump(losses, open(args.loss_file_name, "w"))

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
            import pdb; pdb.set_trace()
            loss = loss_function(prediction, ground_truth)
            loss.backward()
            optimizer.step()

            current_loss = loss.cpu().detach().numpy()

            print(f"Current train loss at epoch {epoch_number} and iteration {index} is {current_loss:.4f}")
            train_loss += current_loss

        epoch_train_loss = train_loss / index

        losses = json.load(open(args.loss_file_name))
        losses['train'].append(epoch_train_loss)
        json.dump(losses, open(args.loss_file_name, "w"))

        torch.save(model.state_dict(), f'./epoch{epoch_number}.pth')

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

        epoch_val_loss = val_loss / index

        losses = json.load(open(args.loss_file_name))
        losses['val'].append(epoch_val_loss)
        json.dump(losses, open(args.loss_file_name, "w"))






