from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .confidence_evaluator import ConfidenceEvaluator
from .run_training import run_training, get_dataset_splits, get_atomistic_splits, evaluate_models
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs

def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)

        # Load the data
        (train_data, val_data, test_data), features_scaler, scaler = \
            get_dataset_splits(args.data_path, args, logger)

        # Train with the data, return the best models
        models = run_training(
            train_data, val_data, scaler, features_scaler, args, logger)

        # Evaluate the models on both val and test data
        val_results = evaluate_models(
            models, train_data, val_data, scaler, args, logger, export_std=True)
        test_results = evaluate_models(
            models, train_data, test_data, scaler, args, logger, export_std=True)

        # Unpack results
        val_scores, val_preds, val_conf, val_std = val_results[:4]
        test_scores, test_preds, test_conf, test_std = test_results[:4]

        # Save test predictions to CSV
        test_targets = np.array(test_data.targets())
        test_smiles = test_data.smiles()
        results_df = pd.DataFrame({
            'SMILES': test_smiles,
            'True Value': test_targets.flatten(),  # Flatten for single-task
            'Predicted Value': test_preds.flatten()  # Flatten for single-task
        })
        results_df.to_csv(os.path.join(args.save_dir, 'test_predictions.csv'), index=False)
        info(f"Test predictions saved to {os.path.join(args.save_dir, 'test_predictions.csv')}")

        # Calculate additional metrics (MAE, MSE, R²)
        mae = mean_absolute_error(test_targets, test_preds)
        mse = mean_squared_error(test_targets, test_preds)
        r2 = r2_score(test_targets, test_preds)
        info(f'Test MAE = {mae:.6f}, Test MSE = {mse:.6f}, Test R² = {r2:.6f}')

        # Log the confidence plots if desired
        if args.confidence:
            val_entropy = val_results[4] if len(val_results) > 4 else None
            test_entropy = test_results[4] if len(test_results) > 4 else None

            ConfidenceEvaluator.save(
                val_preds, val_data.targets(), val_conf, val_std, val_data.smiles(),
                test_preds, test_data.targets(), test_conf, test_std, test_data.smiles(),
                val_entropy, test_entropy, args)

            ConfidenceEvaluator.visualize(
                args.save_confidence, args.confidence_evaluation_methods,
                info, args.save_dir, draw=False)

        all_scores.append(test_scores)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score


def cross_validate_atomistic(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = "U0"  # Atomistic task name

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)

        # Load the data
        (train_data, val_data, test_data), features_scaler, scaler = \
            get_atomistic_splits(args.data_path, args, logger)

        # Train with the data, return the best models
        models = run_training(
            train_data, val_data, scaler, features_scaler, args, logger)

        # Evaluate the models on both val and test data
        val_scores, val_preds, val_conf, val_std, val_entropy = evaluate_models(
            models, train_data, val_data, scaler, args, logger, export_std=True)
        test_scores, test_preds, test_conf, test_std, test_entropy = evaluate_models(
            models, train_data, test_data, scaler, args, logger, export_std=True)

        # Save test predictions to CSV
        test_targets = test_data.targets()
        test_smiles = test_data.smiles()
        results_df = pd.DataFrame({
            'SMILES': test_smiles,
            'True Value': test_targets.flatten(),  # Flatten for single-task
            'Predicted Value': test_preds.flatten()  # Flatten for single-task
        })
        results_df.to_csv(os.path.join(args.save_dir, 'test_predictions.csv'), index=False)
        info(f"Test predictions saved to {os.path.join(args.save_dir, 'test_predictions.csv')}")

        # Calculate additional metrics (MAE, MSE, R²)
        mae = mean_absolute_error(test_targets, test_preds)
        mse = mean_squared_error(test_targets, test_preds)
        r2 = r2_score(test_targets, test_preds)
        info(f'Test MAE = {mae:.6f}, Test MSE = {mse:.6f}, Test R² = {r2:.6f}')

        # Log the confidence plots if desired
        if args.confidence:
            ConfidenceEvaluator.save(
                val_preds, val_data.targets(), val_conf, val_std, val_data.smiles(),
                test_preds, test_data.targets(), test_conf, test_std, test_data.smiles(),
                val_entropy, test_entropy, args)

            ConfidenceEvaluator.visualize(
                args.save_confidence, args.confidence_evaluation_methods,
                info, args.save_dir, draw=False)

        all_scores.append(test_scores)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            info(f'Seed {init_seed + fold_num} ==> test {task_names} {args.metric} = {scores:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    return mean_score, std_score