"""Implements Learning without Forgetting (LwF)."""
from __future__ import division, print_function

import argparse
import copy
import json
import warnings

import dataset
import networks as net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchnet as tnt
import utils as utils
from prune import SparsePruner
from torch.autograd import Variable
from tqdm import tqdm

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode',
                   choices=['finetune', 'eval'],
                   help='Run mode')
FLAGS.add_argument('--finetune_layers',
                   choices=['all', 'fc', 'classifier'], default='all',
                   help='Which layers to finetune')
FLAGS.add_argument('--num_outputs', type=int, default=-1,
                   help='Num outputs for dataset')
# Optimization options.
FLAGS.add_argument('--lr', type=float,
                   help='Learning rate')
FLAGS.add_argument('--lr_decay_every', type=int,
                   help='Step decay every this many epochs')
FLAGS.add_argument('--lr_decay_factor', type=float,
                   help='Multiply lr by this much every step of decay')
FLAGS.add_argument('--finetune_epochs', type=int,
                   help='Number of initial finetuning epochs')
FLAGS.add_argument('--batch_size', type=int, default=32,
                   help='Batch size')
FLAGS.add_argument('--dropout', type=float, default=0.5,
                   help='Dropout ratio')
FLAGS.add_argument('--weight_decay', type=float, default=5e-4,
                   help='Weight decay')
FLAGS.add_argument('--temperature', type=float, default=2.0,
                   help='LwF logit temperature')
FLAGS.add_argument('--ft_shared_after', type=int, default=5,
                   help='Finetune shared layers after this epoch')
# Paths.
FLAGS.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
FLAGS.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
FLAGS.add_argument('--test_path', type=str, default='',
                   help='Location of test data')
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/',
                   help='Location to save model')
FLAGS.add_argument('--loadname', type=str, default='',
                   help='Location to save model')
# Other.
FLAGS.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')


def distillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * scale


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, original_model, model, dataset2idx):
        self.args = args
        self.cuda = args.cuda
        self.dataset2idx = dataset2idx
        self.original_model = original_model
        self.model = model

        if args.mode != 'check':
            # Set up data loader, criterion, and pruner.
            if 'cropped' in args.train_path:
                train_loader = dataset.train_loader_cropped
                test_loader = dataset.test_loader_cropped
            else:
                train_loader = dataset.train_loader
                test_loader = dataset.test_loader
            self.train_data_loader = train_loader(
                args.train_path, args.batch_size, pin_memory=args.cuda)
            self.test_data_loader = test_loader(
                args.test_path, args.batch_size, pin_memory=args.cuda)
            self.criterion = nn.CrossEntropyLoss()

    def eval(self):
        """Performs evaluation."""
        self.model.eval()
        error_meter = None

        print('Performing eval...')
        for batch, label in tqdm(self.test_data_loader, desc='Eval'):
            if self.cuda:
                batch = batch.cuda()
            batch = Variable(batch, volatile=True)

            output = self.model(batch)

            # Init error meter.
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)

        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))
        self.model.train()
        return errors

    def do_batch(self, optimizer, batch, label, epoch_idx):
        """Runs model for one batch."""
        batch_original = batch.clone()
        if self.cuda:
            batch_original = batch_original.cuda(1)
            batch = batch.cuda()
            label = label.cuda()
        batch_original = Variable(batch_original, requires_grad=False)
        batch = Variable(batch)
        label = Variable(label)

        # Get targets using original model.
        self.original_model.eval()
        x = self.original_model.shared(batch_original)
        target_logits = [classifier(x).data.cpu()
                         for classifier in self.original_model.classifiers]
        # Move to same GPU as current model.
        target_logits = [Variable(item.cuda(), requires_grad=False)
                         for item in target_logits]
        scale = [item.size(-1) for item in target_logits]

        # Work with current model.
        # Set grads to 0.
        self.model.zero_grad()

        # Do forward.
        x = self.model.shared(batch)
        pred_logits = [classifier(x) for classifier in self.model.classifiers]

        # Compute loss.
        dist_loss = 0
        # Apply distillation loss to all old tasks.
        for idx in range(len(target_logits)):
            dist_loss += distillation_loss(
                pred_logits[idx], target_logits[idx], self.args.temperature, scale[idx])
        # Apply cross entropy for current task.
        output = pred_logits[-1]
        new_loss = self.criterion(output, label)
        loss = dist_loss + new_loss

        # Do backward.
        loss.backward()

        if epoch_idx <= self.args.ft_shared_after:
            # Set shared layer gradients to 0 if early epochs.
            for module in self.model.shared.modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)
            # Set old classifier gradients to 0 if early epochs.
            for idx in range(len(self.model.classifiers) - 1):
                module = self.model.classifiers[idx]
                module.weight.grad.data.fill_(0)
                module.bias.grad.data.fill_(0)

        # Update params.
        optimizer.step()

    def do_epoch(self, epoch_idx, optimizer):
        """Trains model for one epoch."""
        for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
            self.do_batch(optimizer, batch, label, epoch_idx)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        base_model = self.model
        if isinstance(self.model, nn.DataParallel):
            base_model = self.model.module

        # Prepare the ckpt.
        ckpt = {
            'epoch': epoch,
            'dataset2idx': self.dataset2idx,
            'accuracy': best_accuracy,
            'errors': errors,
            'args': self.args,
            'model': base_model,
        }

        # Save to file.
        torch.save(ckpt, savename + '.pt')

    def train(self, epochs, optimizer, save=True, savename='', best_accuracy=0):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []

        if self.args.cuda:
            self.model = self.model.cuda()

        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))

            optimizer = utils.step_lr(epoch_idx, self.args.lr, self.args.lr_decay_every,
                                      self.args.lr_decay_factor, optimizer)
            self.model.train()
            self.do_epoch(epoch_idx, optimizer)
            errors = self.eval()
            error_history.append(errors)
            accuracy = 100 - errors[0]  # Top-1 accuracy.

            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'args': vars(self.args),
                }, fout)

            # Save best model, if required.
            if save and accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_accuracy, accuracy))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)

    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))


def main():
    """Do stuff."""
    args = FLAGS.parse_args()

    # Set default train and test path if not provided as input.
    if not args.train_path:
        args.train_path = '../data/%s/train' % (args.dataset)
    if not args.test_path:
        if args.dataset == 'imagenet' or args.dataset == 'places':
            args.test_path = '../data/%s/val' % (args.dataset)
        else:
            args.test_path = '../data/%s/test' % (args.dataset)

    # Load the required model.
    if 'finetune' in args.mode and not args.loadname:
        model = net.ModifiedVGG16()
    else:
        ckpt = torch.load(args.loadname)
        model = ckpt['model']
        dataset2idx = ckpt['dataset2idx']

    original_model = copy.deepcopy(model)

    # Add and set the model dataset.
    model.add_dataset(args.dataset, args.num_outputs)
    model.set_dataset(args.dataset)
    if args.dataset not in dataset2idx:
        idxs = [dataset2idx[key] for key in dataset2idx]
        dataset2idx[args.dataset] = max(idxs) + 1
    if args.cuda:
        model = model.cuda(0)
        if args.mode == 'finetune':
            original_model = original_model.cuda(1)

    # Create the manager object.
    manager = Manager(args, original_model, model, dataset2idx)

    # Perform necessary mode operations.
    if args.mode == 'finetune':
        # Get optimizer with correct params.
        if args.finetune_layers == 'all':
            params_to_optimize = model.parameters()
        elif args.finetune_layers == 'classifier':
            for param in model.shared.parameters():
                param.requires_grad = False
            params_to_optimize = model.classifier.parameters()
        elif args.finetune_layers == 'fc':
            params_to_optimize = []
            # Add fc params.
            for param in model.shared.parameters():
                if param.size(0) == 4096:
                    param.requires_grad = True
                    params_to_optimize.append(param)
                else:
                    param.requires_grad = False
            # Add classifier params.
            for param in model.classifier.parameters():
                params_to_optimize.append(param)
            params_to_optimize = iter(params_to_optimize)
        optimizer = optim.SGD(params_to_optimize, lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
        # Perform finetuning.
        manager.train(args.finetune_epochs, optimizer,
                      save=True, savename=args.save_prefix)
    elif args.mode == 'check':
        # Load model and make sure everything is fine.
        manager.check(verbose=True)
    elif args.mode == 'eval':
        # Just run the model on the eval set.
        manager.eval()


if __name__ == '__main__':
    main()
