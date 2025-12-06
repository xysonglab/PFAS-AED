import logging
import functools
import math
import os
from typing import Callable, List, Tuple, Union
from argparse import Namespace

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from chemprop.data import StandardScaler, MoleculeDataset
from chemprop.models import build_model, MoleculeModel
from chemprop.nn_utils import NoamLR, PlateauScheduler


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str,
                    model: MoleculeModel,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None,
                    args: Namespace = None):
    """
    Saves a model checkpoint.

    :param model: A MoleculeModel.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None
    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    cuda: bool = False,
                    logger: logging.Logger = None,
                    dataset: MoleculeDataset = None) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :param dataset: A QM9 dataset. Hack here to initialize schnetpack model with correct embedding
    :return: The loaded MoleculeModel.
    """
    debug = logger.debug if logger is not None else print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    if current_args is not None:
        args = current_args

    args.cuda = cuda

    # Build model
    model = build_model(args, dataset)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            debug(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    return model


def load_scalers(path: str) -> Tuple[StandardScaler, StandardScaler]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data scaler and the features scaler.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    return scaler, features_scaler


def load_args(path: str) -> Namespace:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The arguments Namespace that the model was trained with.
    """
    return torch.load(path, map_location=lambda storage, loc: storage)['args']


def load_task_names(path: str) -> List[str]:
    """
    Loads the task names a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The task names that the model was trained with.
    """
    return load_args(path).task_names


def negative_log_likelihood(pred_targets, pred_var, targets):
    clamped_var = torch.clamp(pred_var, min=0.00001)
    return torch.log(2 * np.pi * clamped_var) / 2 + (pred_targets - targets) ** 2 / (2 * clamped_var)


# Original evidential regression loss (kept for backward compatibility)
def evidential_loss_new_original(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4):
    """
    Original Deep Evidential Regression negative log likelihood loss + evidential regularizer
    """
    # Calculate NLL loss
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    L_NLL = nll

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg

    loss = L_NLL + lam * (L_REG - epsilon)

    return loss


# NEW: Adaptive Evidential Loss with multiple innovations
def evidential_loss_new(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4,
                        temperature=1.0, use_adaptive_reg=False,
                        use_robust_loss=False, huber_delta=1.0,
                        use_calibration=False, calibration_weight=0.01,
                        use_diversity_reg=False, diversity_weight=0.001):
    """
    Adaptive Deep Evidential Regression Loss with improved numerical stability.

    修改说明：
    1. 添加了更多的数值稳定性检查
    2. 使用更保守的clamp范围
    3. 添加了梯度裁剪
    4. 调整了各项损失的权重平衡
    """

    # 首先确保输入参数的数值稳定性
    v = torch.clamp(v, min=epsilon, max=1e6)
    alpha = torch.clamp(alpha, min=1.0 + epsilon, max=1e6)  # 确保 alpha > 1
    beta = torch.clamp(beta, min=epsilon, max=1e6)

    # 温度缩放 - 添加额外的稳定性检查
    temperature = max(temperature, 0.1)  # 防止温度太小
    v_scaled = torch.clamp(v / temperature, min=epsilon, max=1e6)
    alpha_scaled = torch.clamp(alpha * temperature, min=1.0 + epsilon, max=1e6)
    beta_scaled = torch.clamp(beta * temperature, min=epsilon, max=1e6)

    # 计算基础NLL损失 - 使用更稳定的形式
    twoBlambda = 2 * beta_scaled * (1 + v_scaled)
    twoBlambda = torch.clamp(twoBlambda, min=epsilon)  # 防止log(0)

    # 使用更稳定的对数计算
    log_pi_v = 0.5 * torch.log(torch.clamp(np.pi / v_scaled, min=epsilon))
    log_twoBlambda = torch.log(twoBlambda)

    # 计算误差项，添加数值稳定性
    error_term = v_scaled * (targets - mu) ** 2 + twoBlambda
    error_term = torch.clamp(error_term, min=epsilon)

    nll = log_pi_v \
          - alpha_scaled * log_twoBlambda \
          + (alpha_scaled + 0.5) * torch.log(error_term) \
          + torch.lgamma(torch.clamp(alpha_scaled, min=1.0 + epsilon)) \
          - torch.lgamma(torch.clamp(alpha_scaled + 0.5, min=1.5 + epsilon))

    # 防止NaN和Inf
    nll = torch.where(torch.isnan(nll), torch.zeros_like(nll), nll)
    nll = torch.where(torch.isinf(nll), torch.ones_like(nll) * 10.0, nll)  # 用大值替代inf

    L_NLL = nll

    # 计算预测误差
    error = torch.abs(targets - mu)
    error = torch.clamp(error, max=100.0)  # 防止误差过大

    # Robust loss component (Huber-like) for outlier handling
    if use_robust_loss:
        # 使用更平滑的过渡
        huber_delta = max(huber_delta, 0.1)
        is_small_error = error < huber_delta
        robust_error = torch.where(
            is_small_error,
            0.5 * error ** 2,
            huber_delta * (error - 0.5 * huber_delta)
        )
        robust_error = torch.clamp(robust_error, max=100.0)
    else:
        robust_error = error

    # 置信度自适应正则化
    if use_adaptive_reg:
        # 更稳定的认知不确定性估计
        alpha_safe = torch.clamp(alpha_scaled - 1, min=epsilon)
        epistemic_uncertainty = torch.sqrt(torch.clamp(beta_scaled / alpha_safe, min=epsilon, max=1e6))

        # 自适应权重：使用更平滑的函数
        adaptive_weight = torch.exp(-epistemic_uncertainty / 10.0)  # 使用指数衰减而不是倒数
        adaptive_weight = torch.clamp(adaptive_weight, min=0.01, max=1.0)

        # 应用自适应权重
        reg = adaptive_weight * robust_error * torch.clamp(2 * v_scaled + alpha_scaled, max=1000.0)
    else:
        # 标准正则化项
        reg = robust_error * torch.clamp(2 * v_scaled + alpha_scaled, max=1000.0)

    L_REG = torch.clamp(reg, max=100.0)  # 限制正则化项的最大值

    # 校准项：鼓励预测方差匹配观察误差
    if use_calibration:
        # 更稳定的方差预测
        alpha_safe = torch.clamp(alpha_scaled - 1, min=epsilon)
        predicted_var = torch.clamp(beta_scaled / alpha_safe, min=epsilon, max=1e6)

        # 观察到的平方误差
        squared_error = torch.clamp((targets - mu) ** 2, max=1e6)

        # 使用相对误差而不是绝对误差
        relative_error = torch.abs(torch.log(predicted_var + 1) - torch.log(squared_error + 1))
        L_CALIB = calibration_weight * torch.clamp(relative_error, max=10.0)
    else:
        L_CALIB = 0

    # 多样性正则化：防止不确定性估计崩溃
    if use_diversity_reg:
        # 计算批次内参数的均值
        mean_v = torch.mean(v_scaled)
        mean_alpha = torch.mean(alpha_scaled)
        mean_beta = torch.mean(beta_scaled)

        # 计算方差（越高越好）- 添加稳定性
        var_v = torch.clamp(torch.var(v_scaled), min=0, max=100.0)
        var_alpha = torch.clamp(torch.var(alpha_scaled), min=0, max=100.0)
        var_beta = torch.clamp(torch.var(beta_scaled), min=0, max=100.0)

        # 多样性损失：使用对数形式以提高稳定性
        L_DIVERSITY = -diversity_weight * torch.log(1 + var_v + var_alpha + var_beta)

        # 防止参数变得太小的惩罚项
        min_param_penalty = diversity_weight * (
                torch.relu(epsilon - mean_v) +
                torch.relu(1 + epsilon - mean_alpha) +
                torch.relu(epsilon - mean_beta)
        )

        L_DIVERSITY = torch.clamp(L_DIVERSITY + min_param_penalty, min=-10.0, max=10.0)
    else:
        L_DIVERSITY = 0

    # 组合所有损失项 - 使用更保守的权重
    loss = L_NLL + lam * torch.clamp(L_REG - epsilon, min=0, max=100.0)

    # 逐步添加其他项
    if use_calibration:
        loss = loss + L_CALIB
    if use_diversity_reg:
        loss = loss + L_DIVERSITY

    # 最终的稳定性检查
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.where(torch.isinf(loss), torch.ones_like(loss) * 100.0, loss)
    loss = torch.clamp(loss, min=0, max=1000.0)  # 限制损失的范围

    return loss


# evidential regression (original, kept for compatibility)
def evidential_loss(mu, v, alpha, beta, targets):
    """
    Use Deep Evidential Regression Sum of Squared Error loss

    :mu: Pred mean parameter for NIG
    :v: Pred lambda parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """

    # Calculate SOS
    # Calculate gamma terms in front
    def Gamma(x):
        return torch.exp(torch.lgamma(x))

    coeff_denom = 4 * Gamma(alpha) * v * torch.sqrt(beta)
    coeff_num = Gamma(alpha - 0.5)
    coeff = coeff_num / coeff_denom

    # Calculate target dependent loss
    second_term = 2 * beta * (1 + v)
    second_term += (2 * alpha - 1) * v * torch.pow((targets - mu), 2)
    L_SOS = coeff * second_term

    # Calculate regularizer
    L_REG = torch.pow((targets - mu), 2) * (2 * alpha + v)

    loss_val = L_SOS + L_REG

    return loss_val


def get_loss_func(args: Namespace) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Namespace containing the dataset type (regression only).
    :return: A PyTorch loss function.
    """
    if args.confidence == 'nn':
        return negative_log_likelihood

    # Allow testing of both of these loss functions
    if args.confidence == 'evidence' and args.new_loss:
        # Use adaptive evidential loss with configurable parameters
        return functools.partial(
            evidential_loss_new,
            lam=args.regularizer_coeff,
            temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
            use_adaptive_reg=args.use_adaptive_reg if hasattr(args, 'use_adaptive_reg') else True,
            use_robust_loss=args.use_robust_loss if hasattr(args, 'use_robust_loss') else True,
            huber_delta=args.huber_delta if hasattr(args, 'huber_delta') else 1.0,
            use_calibration=args.use_calibration if hasattr(args, 'use_calibration') else True,
            calibration_weight=args.calibration_weight if hasattr(args, 'calibration_weight') else 0.1,
            use_diversity_reg=args.use_diversity_reg if hasattr(args, 'use_diversity_reg') else True,
            diversity_weight=args.diversity_weight if hasattr(args, 'diversity_weight') else 0.01
        )
    if args.confidence == 'evidence':
        return evidential_loss

    if args.metric == "rmse":
        return nn.MSELoss(reduction='none')
    elif args.metric == "mae":
        return nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"metric {args.metric} must be rmse, mae, or r2 for regression")


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    raise ValueError(f'Metric "{metric}" not supported.')


def build_optimizer(model: nn.Module, args: Namespace) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    params = [{'params': model.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]

    return Adam(params)


def build_lr_scheduler(optimizer: Optimizer, args: Namespace,
                       total_epochs: List[int] = None,
                       scheduler_name: str = "noam",
                       ) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :param scheduler_name: Name of scheduler
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    if scheduler_name == "plateau":
        return PlateauScheduler(optimizer=optimizer, patience=args.patience,
                                factor=args.factor, final_lr=args.final_lr)
    elif scheduler_name == "noam":
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=[args.warmup_epochs],
            total_epochs=total_epochs or [args.epochs] * args.num_lrs,
            steps_per_epoch=args.train_data_size // args.batch_size,
            init_lr=[args.init_lr],
            max_lr=[args.max_lr],
            final_lr=[args.final_lr]
        )
    else:
        raise NotImplementedError(f"Scheduler name {scheduler_name} is not implemented")


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger