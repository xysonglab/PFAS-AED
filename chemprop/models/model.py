from argparse import Namespace
import logging
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class EvaluationDropout(nn.Dropout):
    def __init__(self, *args, **kwargs):
        super(EvaluationDropout, self).__init__(*args, **kwargs)
        self.inference_mode = False

    def set_inference_mode(self, val: bool):
        self.inference_mode = val

    def forward(self, input):
        if self.inference_mode:
            return nn.functional.dropout(input, p=0)
        else:
            return nn.functional.dropout(input, p=self.p)


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, confidence: bool = False, conf_type: Optional[str] = None):
        """
        Initializes the MoleculeModel.

        :param confidence: Whether confidence values should be predicted.
        :param conf_type: Str definition of what type of confidence to use
        """
        super(MoleculeModel, self).__init__()

        # NOTE: Confidence flag is only set if the model must handle returning
        # confidence internally and for evidential learning.
        self.confidence = confidence
        self.conf_type = conf_type

        self.use_last_hidden = True

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)
        self.args = args

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        first_linear_dim = args.hidden_size
        if args.use_input_features:
            first_linear_dim += args.features_dim

        # When using dropout for confidence, use dropouts for evaluation in addition to training.
        if args.confidence == 'dropout':
            self.dropout = EvaluationDropout(args.dropout)
        else:
            self.dropout = nn.Dropout(args.dropout)

        activation = get_activation_function(args.activation)

        output_size = args.output_size

        if self.confidence:  # if confidence should be learned
            if args.confidence == 'evidence':
                # normal inverse gamma
                # For each task, output the parameters of the NIG
                # distribution (gamma, lambda, alpha, beta)
                output_size *= 4
            else:  # gaussian MVE
                # For each task output the paramters of the Normal
                # distribution (mu, var)
                output_size *= 2

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                self.dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            ffn = [
                self.dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 3):
                ffn.extend([
                    activation,
                    self.dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])

            ffn.extend([
                activation,
                self.dropout,
                nn.Linear(args.ffn_hidden_size, args.last_hidden_size),
            ])

            ffn.extend([
                activation,
                self.dropout,
                nn.Linear(args.last_hidden_size, output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        ffn = self.ffn if self.use_last_hidden else nn.Sequential(
            *list(self.ffn.children())[:-1])
        output = ffn(self.encoder(*input))

        if self.confidence:
            if self.conf_type == "evidence":
                min_val = 1e-6

                # Split the outputs into the four distribution parameters
                means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1] // 4, dim=1)
                lambdas = torch.nn.Softplus()(loglambdas) + min_val
                alphas = torch.nn.Softplus()(
                    logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
                betas = torch.nn.Softplus()(logbetas) + min_val

                # Return these parameters as the output of the model
                output = torch.stack((means, lambdas, alphas, betas),
                                     dim=2).view(output.size())
            else:  # MVE (Mean-Variance Estimation)
                even_indices = torch.tensor(range(0, list(output.size())[1], 2))
                odd_indices = torch.tensor(range(1, list(output.size())[1], 2))

                if self.args.cuda:
                    even_indices = even_indices.cuda()
                    odd_indices = odd_indices.cuda()

                predicted_means = torch.index_select(output, 1, even_indices)
                predicted_confidences = torch.index_select(output, 1, odd_indices)
                capped_confidences = nn.functional.softplus(predicted_confidences)

                output = torch.stack((predicted_means, capped_confidences), dim=2).view(output.size())

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size

    if args.confidence == 'nn':
        model = MoleculeModel(confidence=True, conf_type="nn")
    elif args.confidence == "evidence":
        model = MoleculeModel(confidence=True, conf_type="evidence")
    else:
        model = MoleculeModel()

    model.create_encoder(args)
    model.create_ffn(args)
    initialize_weights(model)

    return model