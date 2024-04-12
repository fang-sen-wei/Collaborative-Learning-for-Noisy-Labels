import torch
import argparse
import util
import os
import datetime
import random
import mlconfig
import loss
import models
import dataset
import shutil
from evaluator import Evaluator
from trainer import Trainer
import numpy as np

# ArgParse
parser = argparse.ArgumentParser(description='Normalized Loss Functions for Deep Learning with Noisy Labels')
# Training
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config_path', type=str, default='configs')
parser.add_argument('--version', type=str, default='ce')
parser.add_argument('--exp_name', type=str, default="run1")
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--asym', action='store_true', default=False)
parser.add_argument('--noise_rate', type=float, default=0.0)
parser.add_argument('--num_gradual', type=int, default=10)
parser.add_argument('--exponent', type=float, default=1)
parser.add_argument('--data_type', type=str, default="cifar10")

args = parser.parse_args()

# Set up
if args.exp_name == '' or args.exp_name is None:
    args.exp_name = 'exp_' + datetime.datetime.now()
exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)

# log file
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")
for arg in vars(args):
    logger.info("%s: %s" % (arg, getattr(args, arg)))

random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    logger.info("Using CUDA!")
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

logger.info("PyTorch Version: %s" % (torch.__version__))
config_file =os.path.join(args.config_path, args.version) + '.yaml'
print(config_file)
config = mlconfig.load(config_file)
print(config)
config.set_immutable()
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))
for key in config:
    logger.info("%s: %s" % (key, config[key]))

# dropout rate limitation
# for cifar100 e=0.1, for mnist\cifar10 e=0.4
if args.data_type == "cifar100":
    e = 0.1
    # e represents the \omega in the paper
else:
    e = 0.4

if args.noise_rate < e:
    forget_rate = args.noise_rate
else:
    forget_rate = e

# define drop rate schedule
rate_schedule = np.ones(config.epochs)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)


def main():
    if config.dataset.name == 'DatasetGenerator':

        data_loader = config.dataset(seed=args.seed, noise_rate=args.noise_rate, asym=args.asym)
    else:
        data_loader = config.dataset()

    model1 = config.model()
    model2 = config.model()
    model1 = model1.to(device)
    model2 = model2.to(device)

    data_loader = data_loader.getDataLoader()
    logger.info("param size = %fMB", util.count_parameters_in_MB(model1))
    if args.data_parallel:
        model1 = torch.nn.DataParallel(model1)
        model2 = torch.nn.DataParallel(model2)

    optimizer1 = config.optimizer(model1.parameters())
    optimizer2 = config.optimizer(model2.parameters())
    scheduler1 = config.scheduler(optimizer1)
    scheduler2 = config.scheduler(optimizer2)
    if config.criterion.name == 'NLNL':
        criterion1 = config.criterion(train_loader=data_loader['train_dataset'])
        criterion2 = config.criterion(train_loader=data_loader['train_dataset'])
    else:
        criterion1 = config.criterion()
        criterion2 = config.criterion()

    # Initialize Trainer and Evaluator
    trainer = Trainer(data_loader['train_dataset'], logger, config)
    evaluator = Evaluator(data_loader['test_dataset'], logger, config)

    starting_epoch = 0
    ENV_1 = {'global_step': 0,
             'best_acc': 0.0,
             'current_acc': 0.0,
             'train_history': [],
             'eval_history': []}

    ENV_2 = {'global_step': 0,
             'best_acc': 0.0,
             'current_acc': 0.0,
             'train_history': [],
             'eval_history': []}

    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file + "_model1",
                                     model=model1,
                                     optimizer=optimizer1,
                                     scheduler=scheduler1)
        starting_epoch = checkpoint['epoch']
        ENV_1 = checkpoint['ENV']
        trainer.global_step = ENV_1['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file + " model1"))

        checkpoint = util.load_model(filename=checkpoint_path_file + "_model2",
                                     model=model2,
                                     optimizer=optimizer2,
                                     scheduler=scheduler2)
        starting_epoch = checkpoint['epoch']
        ENV_2 = checkpoint['ENV']
        trainer.global_step = ENV_2['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file + " model2"))

    train(starting_epoch, model1, model2, optimizer1, optimizer2, scheduler1, scheduler2, criterion1, criterion2, trainer, evaluator, ENV_1, ENV_2)
    return

def train(starting_epoch, model1, model2, optimizer1, optimizer2, scheduler1, scheduler2, criterion1, criterion2, trainer, evaluator, ENV_1, ENV_2):
    for epoch in range(starting_epoch, config.epochs):
        logger.info("=" * 20 + "Training" + "=" * 20)
        GLOBAL_STEP = trainer.train(epoch, ENV_1['global_step'], model1, criterion1, optimizer1, model2, criterion2, optimizer2, rate_schedule[epoch])
        ENV_1['global_step'] = GLOBAL_STEP
        ENV_2['global_step'] = GLOBAL_STEP
        scheduler1.step()
        scheduler2.step()

        # Eval model1
        logger.info("=" * 20 + "Eval_model1" + "=" * 20)
        evaluator.eval(epoch, ENV_1['global_step'], model1, torch.nn.CrossEntropyLoss())
        acc1 = evaluator.acc_meters.avg * 100
        payload = ('Eval model_1 Loss:%.4f\tEval model1 acc: %.2f' % (
            evaluator.loss_meters.avg, evaluator.acc_meters.avg * 100))
        logger.info(payload)
        ENV_1['train_history'].append(trainer.acc_meters.avg * 100)
        ENV_1['eval_history'].append(evaluator.acc_meters.avg * 100)
        ENV_1['current_acc'] = evaluator.acc_meters.avg * 100
        ENV_1['best_acc'] = max(ENV_1['current_acc'], ENV_1['best_acc'])

        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Eval model2
        logger.info("=" * 20 + "Eval_model2" + "=" * 20)
        evaluator.eval(epoch, ENV_2['global_step'], model2, torch.nn.CrossEntropyLoss())
        acc2 = evaluator.acc_meters.avg * 100
        payload = ('Eval model_2 Loss:%.4f\tEval model2 acc: %.2f' % (
            evaluator.loss_meters.avg, evaluator.acc_meters.avg * 100))
        logger.info(payload)
        ENV_2['train_history'].append(trainer.acc_meters.avg * 100)
        ENV_2['eval_history'].append(evaluator.acc_meters.avg * 100)
        ENV_2['current_acc'] = evaluator.acc_meters.avg * 100
        ENV_2['best_acc'] = max(ENV_2['current_acc'], ENV_2['best_acc'])

        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Print accuracies
        logger.info("Accuracy of model1: %.2f%%" % acc1)
        logger.info("Accuracy of model2: %.2f%%" % acc2)

        # Save Model
        target_model1 = model1.module if args.data_parallel else model1
        target_model2 = model2.module if args.data_parallel else model2
        util.save_model(ENV=ENV_1,
                        epoch=epoch,
                        model=target_model1,
                        optimizer=optimizer1,
                        scheduler=scheduler1,
                        filename=checkpoint_path_file + "_model1")
        util.save_model(ENV=ENV_2,
                        epoch=epoch,
                        model=target_model2,
                        optimizer=optimizer2,
                        scheduler=scheduler2,
                        filename=checkpoint_path_file + "_model2")
        logger.info('Models Saved at %s and %s', checkpoint_path_file + "_model1", checkpoint_path_file + "_model2")

    return

if __name__ == '__main__':
    main()
