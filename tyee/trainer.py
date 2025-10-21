#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : trainer.py
@Time    : 2024/09/25 16:46:17
@Desc    : 
"""
import os
import math
import torch
import logging
import torch.distributed as dist
from tyee.utils import MetricEvaluator
from torch.utils.tensorboard import SummaryWriter
from tyee.utils import lazy_import_module, get_nested_field, get_grad_norm, format_value
from torch.nn.parallel import DistributedDataParallel as DDP

INT_MAX = 2**31 - 1

class Trainer:
    """
    Trainer class for managing the training, evaluation, and checkpointing process.

    Args:
        cfg (dict): Configuration dictionary.
        rank (int): Rank of the current process in distributed training.
        world_size (int): Total number of processes in distributed training.
    """

    def __init__(self, cfg, rank, world_size) -> None:
        """
        Initialize the Trainer class.

        Args:
            cfg (dict): Configuration dictionary.
            rank (int): Rank of the current process in distributed training.
            world_size (int): Total number of processes in distributed training.
        """
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        # Task configuration
        self.task_select = get_nested_field(self.cfg, 'task.select', '')
        self.task = self._build_task()

        # Mixed precision configuration
        self.fp16 = get_nested_field(self.cfg, 'trainer.fp16', False)

        # Training configuration
        self._total_steps = get_nested_field(self.cfg, 'trainer.total_steps', INT_MAX)
        self._total_epochs = get_nested_field(self.cfg, 'trainer.total_epochs', INT_MAX)
        self._update_interval = get_nested_field(self.cfg, 'trainer.update_interval', 1)
        print(f"Update interval: {self._update_interval}")
        self._save_interval = get_nested_field(self.cfg, 'trainer.save_interval', None)
        self._eval_interval = get_nested_field(self.cfg, 'trainer.eval_interval', None)
        self._log_interval = get_nested_field(self.cfg, 'trainer.log_interval', 16)
        self._batch_size = get_nested_field(self.cfg, 'dataset.batch_size', 1)
        self._max_grad_norm = get_nested_field(self.cfg, 'trainer.max_grad_norm', None)

        # Metrics configuration
        metrics_list = get_nested_field(self.cfg, 'trainer.metrics', None)
        self.eval_metric = get_nested_field(self.cfg, 'trainer.eval_metric.select', metrics_list[0])
        self.eval_metric_mode = get_nested_field(self.cfg, 'trainer.eval_metric.mode', 'max')
        self._metrics_evaluator = MetricEvaluator(metrics_list)

        # Experiment directories
        self.exp_dir = get_nested_field(self.cfg, "common.exp_dir")
        self.tb_root = get_nested_field(self.cfg, "common.tb_dir")
        self.checkpoint_root = get_nested_field(self.cfg, "common.checkpoint_dir")

    def run(self):
        """
        Execute the training process, including initialization, training, evaluation, and checkpointing.
        """
        for fold, (train_dataset, val_dataset, eval_dataset) in enumerate(self.task.get_datasets()):
            self._init_components(fold, train_dataset, val_dataset, eval_dataset)
            self.logger.info(f"Start training for fold {fold + 1}")
            self.run_loop()
            if self.best_metric_value is not None:
                self.logger.info(f"Best {self.eval_metric}: {self.best_metric_value}")

    def _build_task(self) -> object:
        """
        Dynamically import and build the task class.

        Returns:
            object: An instance of the task class.
        """
        module_name, class_name = self.task_select.rsplit('.', 1)
        task = lazy_import_module(f'tasks.{module_name}', class_name)
        return task(self.cfg)

    def _init_components(self, fold, train_dataset, val_dataset, test_dataset):
        """
        Initialize components for training, including logging, datasets, and model.

        Args:
            fold (int): Current fold index.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            test_dataset (Dataset): Test dataset.
        """
        # Logging configuration
        self.logger = logging.getLogger(__name__)

        # Checkpoint directory
        self.checkpoint_dir = os.path.join(self.checkpoint_root, f'fold_{fold}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.last_checkpoint = None

        # TensorBoard directory
        tb_fold_dir = os.path.join(self.tb_root, f'fold_{fold}')
        os.makedirs(tb_fold_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_fold_dir)

        # Dataset configuration
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Samplers
        self.train_sampler = self.task.build_sampler(self.train_dataset, self.world_size, self.rank)
        self.val_sampler = self.task.build_sampler(self.val_dataset, self.world_size, self.rank) if self.val_dataset else None
        self.test_sampler = self.task.build_sampler(self.test_dataset, self.world_size, self.rank) if self.test_dataset else None

        # Data loaders
        self.train_loader = self.task.build_dataloader(self.train_dataset, self._batch_size, sampler=self.train_sampler, shuffle=True)
        self.val_loader = self.task.build_dataloader(self.val_dataset, self._batch_size, sampler=self.val_sampler, shuffle=False) if self.val_dataset else None
        self.test_loader = self.task.build_dataloader(self.test_dataset, self._batch_size, sampler=self.test_sampler, shuffle=False) if self.test_dataset else None

        # Training steps and intervals
        niter_per_epoch = math.ceil(len(self.train_dataset) / (self._batch_size * self.world_size))
        self._total_steps = min(self._total_steps, self._total_epochs * niter_per_epoch)
        self._eval_interval = self._eval_interval or niter_per_epoch
        self._save_interval = self._save_interval or niter_per_epoch

        # Model, optimizer, and scheduler
        self.device = torch.device(self.rank)
        self.model = self.task.build_model().to(self.device)
        self.optimizer = self.task.build_optimizer(self.model)
        self.lr_scheduler = self.task.build_lr_scheduler(self.optimizer, niter_per_epoch)
        self.start_step = 1
        self.best_metric_value = None

        # Distributed training
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])

        # Mixed precision training
        self.scaler = torch.amp.GradScaler(enabled=self.fp16) if self.fp16 else None

        # Resume from checkpoint
        self.resume_enabled = get_nested_field(self.cfg, 'trainer.resume.enabled', False)
        if self.resume_enabled:
            resume_checkpoint = get_nested_field(self.cfg, 'trainer.resume.checkpoint', '')
            if not os.path.exists(resume_checkpoint):
                raise FileNotFoundError(f"Resume checkpoint file {resume_checkpoint} does not exist!")
            self._load_checkpoint(resume_checkpoint)
            self.logger.info(f"Model loaded from checkpoint {resume_checkpoint}")
 
    def run_loop(self):
        """
        Run the training and evaluation loop, including logging and checkpointing.
        """
        iterator = self.task.get_batch_iterator(self.train_loader)
        self.task.on_train_epoch_start(self, self.start_step)
        # Skip to the current step's batch
        for _ in range((self.start_step - 1) % len(self.train_loader)):
            sample = self.task.load_sample(iterator, self.device)

        # Initialize accumulated metrics
        train_metrics = {}
        val_metrics = {}
        test_metrics = {}
        self.task.on_train_start(self)
        for step in range(self.start_step, self._total_steps + 1):
            # Update epoch and iterator
            if step % len(self.train_loader) == 0:
                self.task.on_train_epoch_end(self, step-1)
                if self.train_sampler is not None:
                    # Update epoch for distributed training
                    self.train_sampler.set_epoch(step // len(self.train_loader))
                iterator = self.task.get_batch_iterator(self.train_loader)
                self.task.on_train_epoch_start(self, step)

            # Perform a single training step
            # train_result = {}
            train_result = self._train_step(iterator, step, val_metrics)

            # Accumulate training metrics
            for metric, value in train_result.items():
                train_metrics[metric] = train_metrics.get(metric, 0) + value

            # Compute gradient norm before updating parameters
            grad_norm = get_grad_norm(self.model.parameters(), norm_type=2)

            # Log optimizer-related metrics
            if self.rank == 0:
                min_lr, max_lr = self._log_optimizer_metrics(step, grad_norm)
                train_metrics['min_lr'] = train_metrics.get('min_lr', 0) + min_lr
                train_metrics['max_lr'] = train_metrics.get('max_lr', 0) + max_lr

            # Log training metrics
            if step % self._log_interval == 0 and self.rank == 0:
                self._log_training_metrics(step, train_metrics)
                train_metrics = {}

            # Evaluate on validation and test sets
            if step % self._eval_interval == 0:
                val_metrics, test_metrics = self._evaluate(step)

            # Save checkpoints
            if step % self._save_interval == 0 and self.rank == 0:
                self._save_checkpoint_step(step)
        self.task.on_train_end(self)

    def _train_step(self, iterator, step, val_metrics):
        """
        Perform a single training step, including forward pass, loss computation, backward pass, and parameter update.

        Args:
            iterator: Iterator for the current batch.
            step (int): Current training step.

        Returns:
            dict: A dictionary containing loss and computed metrics.
        """
        sample = self.task.load_sample(iterator, self.device)
        self.model.train()
        # Use automatic mixed precision (AMP)
        with torch.amp.autocast('cuda', enabled=self.fp16):
            result = self.task.train_step(self.model, sample)
            loss = result['loss'].clone()
            loss /= self._update_interval

        # Backward pass
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update parameters every `update_interval` steps
        if step % self._update_interval == 0:
            self._update_parameters()

        # Update learning rate scheduler
        if self.lr_scheduler:
            metric = None
            if self.lr_scheduler.metric_source == 'train':
                metric = result.get(self.lr_scheduler.metric, None)
                if isinstance(metric, torch.Tensor):
                    metric = metric.item()
            elif self.lr_scheduler.metric_source == 'val':
                metric = val_metrics.get(self.lr_scheduler.metric, None)
                if isinstance(metric, torch.Tensor):
                    metric = metric.item()
            self.lr_scheduler.step(metric, step)
        
        result = self._aggregate_metrics(result['loss'].item(), metrics= None)
        return result

    def _update_parameters(self):
        """
        Update model parameters, including gradient clipping and optimizer step.
        """
        if self.fp16:
            self.scaler.unscale_(self.optimizer)
            if self._max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self._max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._max_grad_norm)
            self.optimizer.step()

        self.optimizer.zero_grad()

    def _log_optimizer_metrics(self, step, grad_norm):
        """
        Log optimizer-related metrics, such as learning rate and gradient norm.

        Args:
            step (int): Current training step.
            grad_norm (float): Gradient norm.
        """
        min_lr, max_lr, weight_decay = float('inf'), 0.0, 0.0
        for group in self.optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
            weight_decay = group['weight_decay']

        self.writer.add_scalar('opt/lr', max_lr, step)
        self.writer.add_scalar('opt/min_lr', min_lr, step)
        self.writer.add_scalar('opt/weight_decay', weight_decay, step)
        self.writer.add_scalar('opt/grad_norm', grad_norm, step)

        return min_lr, max_lr

    def _log_training_metrics(self, step, accumulated_train_metrics):
        """
        Log training metrics to the console and TensorBoard.

        Args:
            step (int): Current training step.
            accumulated_train_metrics (dict): Accumulated training metrics.
        """
        avg_train_metrics = {
            metric: format_value(value / self._log_interval) 
            for metric, value in accumulated_train_metrics.items()
        }

        self.logger.info(f"Train, Step {step}, Metrics: {avg_train_metrics}")
        for metric, value in avg_train_metrics.items():
            self.writer.add_scalar(f'train/{metric}', float(value), step)
    
    def _evaluate(self, step):
        """
        Evaluate the model on validation and test sets and log the results.

        Args:
            step (int): Current training step.
        """
        val_result, test_result = {}, {}
        if self.val_loader is not None:
            val_result = self._eval_step(self.val_loader)
        if self.test_loader is not None:
            test_result = self._eval_step(self.test_loader)

        if self.rank == 0:
            if self.val_loader is not None:
                formatted_val_result = {
                    metric: format_value(value) for metric, value in val_result.items()
                }
                self.logger.info(f"Val, Step {step}, Metrics: {formatted_val_result}")
                for metric, value in formatted_val_result.items():
                    self.writer.add_scalar(f'val/{metric}', float(value), step)

            if self.test_loader is not None:
                formatted_test_result = {
                    metric: format_value(value) for metric, value in test_result.items()
                }
                self.logger.info(f"Test, Step {step}, Metrics: {formatted_test_result}")
                for metric, value in formatted_test_result.items():
                    self.writer.add_scalar(f'test/{metric}', float(value), step)

            self._check_and_save_best(val_result, step)
        return val_result, test_result

    def _eval_step(self, loader):
        """
        Perform a single evaluation step, including model evaluation, loss computation, and metric calculation.

        Args:
            loader (DataLoader): DataLoader for loading evaluation data.

        Returns:
            dict: A dictionary containing the total evaluation loss and computed metrics.
        """
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            iterator = self.task.get_batch_iterator(loader)
            self.task.on_valid_start(self)
            for _ in range(len(loader)):
                sample = self.task.load_sample(iterator, self.device)
                # Use automatic mixed precision (AMP)
                with torch.amp.autocast('cuda', enabled=self.fp16):
                    result = self.task.valid_step(self.model, sample)
                    total_loss += result['loss'].item()

                # Update metrics
                self._metrics_evaluator.update_metrics(result)

            # Calculate final metrics
            metrics = self._metrics_evaluator.calculate_metrics()

            self.task.on_valid_end(self)
        # Aggregate and average loss and metrics
        result = self._aggregate_metrics(total_loss, metrics, len(loader))
        return result

    def _aggregate_metrics(self, loss, metrics, total=None):
        """
        Aggregate and average loss and metrics across processes (for distributed training).

        Args:
            loss (float or torch.Tensor): Current step loss.
            metrics (dict): Current step metrics.
            total (int, optional): Total number of samples (used for averaging).

        Returns:
            dict: A dictionary containing aggregated and averaged loss and metrics.
        """
        if self.world_size > 1:
            # Aggregate loss across processes
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss).to(self.device)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.world_size

            # Aggregate metrics across processes
            if metrics is not None:
                for metric, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        dist.all_reduce(value, op=dist.ReduceOp.SUM)
                        value /= self.world_size
                        metrics[metric] = value.item()
                    elif isinstance(value, (int, float)):
                        tensor_value = torch.tensor(value).to(self.device)
                        dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)
                        metrics[metric] = tensor_value.item() / self.world_size

        # Average loss over total samples if provided
        if total is not None:
            loss /= total

        result = {"loss": loss}
        if metrics is not None:
            result.update(metrics)

        return result

    def _check_and_save_best(self, val_result, step):
        """
        Check if the current evaluation metric is the best and save the best model checkpoint.

        Args:
            val_result (dict): Dictionary containing the current evaluation results.
            step (int): Current training step.
        """
        current_metric_value = val_result.get(self.eval_metric, None)

        if current_metric_value is not None:
            save_best = False

            # Compare with the best metric value based on eval_metric_mode
            if self.eval_metric_mode == "max":
                if self.best_metric_value is None or current_metric_value > self.best_metric_value:
                    save_best = True
            elif self.eval_metric_mode == "min":
                if self.best_metric_value is None or current_metric_value < self.best_metric_value:
                    save_best = True
            else:
                raise ValueError(f"Invalid eval_metric_mode: {self.eval_metric_mode}. Must be 'max' or 'min'.")

            if save_best:
                self.best_metric_value = current_metric_value
                best_model_filename = f"{self.checkpoint_dir}/checkpoint_best.pt"
                self._save_checkpoint(best_model_filename, self.model, self.optimizer, self.lr_scheduler, step)
                self.logger.info(f"Best checkpoint saved with {self.eval_metric}: {current_metric_value}")

    def _state_dict(self, model, optimizer, lr_scheduler, step):
        """
        Create a state dictionary for saving checkpoints.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            lr_scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler to save.
            step (int): Current training step.

        Returns:
            dict: A dictionary containing the state of the model, optimizer, scheduler, and other metadata.
        """
        state = {
            "args": self.cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_metric_value": self.best_metric_value
        }
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler.state_dict()
        return state

    def _save_checkpoint_step(self, step):
        """
        Save a checkpoint at the current training step.

        Args:
            step (int): Current training step.
        """
        checkpoint_filename = f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"
        self._save_checkpoint(checkpoint_filename, self.model, self.optimizer, self.lr_scheduler, step)

        # Remove the last checkpoint if it exists
        if hasattr(self, 'last_checkpoint') and self.last_checkpoint is not None and os.path.exists(self.last_checkpoint):
            os.remove(self.last_checkpoint)

        self.last_checkpoint = checkpoint_filename
        self.logger.info(f"Checkpoint saved for step {step}")

    def _save_checkpoint(self, filename, model, optimizer, lr_scheduler, step):
        """
        Save a checkpoint to a file.

        Args:
            filename (str): Path to save the checkpoint.
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            lr_scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler to save.
            step (int): Current training step.
        """
        checkpoint = self._state_dict(model, optimizer, lr_scheduler, step)
        torch.save(checkpoint, filename)
        self.logger.info(f"Checkpoint saved to {filename}")

    def _load_checkpoint(self, filename):
        """
        Load a checkpoint from a file.

        Args:
            filename (str): Path to the checkpoint file.
        """
        ckpt_params = torch.load(filename)
        self.cfg = ckpt_params["args"]
        self.model.load_state_dict(ckpt_params["model"])
        self.optimizer.load_state_dict(ckpt_params["optimizer"])
        if "lr_scheduler" in ckpt_params and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt_params["lr_scheduler"])
        self.start_step = ckpt_params["step"] + 1
        self.best_metric_value = ckpt_params["best_metric_value"]
        self.logger.info(f"Checkpoint loaded from {filename}")