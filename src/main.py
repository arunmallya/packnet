"""Main entry point for doing all pruning-related stuff."""
from __future__ import division, print_function

import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from tqdm import tqdm

import dataset
import networks as net
import utils as utils
from prune import SparsePruner


# To prevent PIL warnings.
warnings.filterwarnings("ignore")

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--arch',
                   choices=['vgg16', 'vgg16bn', 'resnet50', 'densenet121'],
                   help='Architectures')
FLAGS.add_argument('--mode',
                   choices=['finetune', 'prune', 'check', 'eval'],
                   help='Run mode')
FLAGS.add_argument('--finetune_layers',
                   choices=['all', 'fc', 'classifier'], default='all',
                   help='Which layers to finetune, fc only works with vgg')
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
FLAGS.add_argument('--weight_decay', type=float, default=0.0,
                   help='Weight decay')
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
# Pruning options.
FLAGS.add_argument('--prune_method', type=str, default='sparse',
                   choices=['sparse'],
                   help='Pruning method to use')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.5,
                   help='% of neurons to prune per layer')
FLAGS.add_argument('--post_prune_epochs', type=int, default=0,
                   help='Number of epochs to finetune for after pruning')
FLAGS.add_argument('--disable_pruning_mask', action='store_true', default=False,
                   help='use masking or not')
FLAGS.add_argument('--train_biases', action='store_true', default=False,
                   help='use separate biases or not')
FLAGS.add_argument('--train_bn', action='store_true', default=False,
                   help='train batch norm or not')
# Other.
FLAGS.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')
FLAGS.add_argument('--init_dump', action='store_true', default=False,
                   help='Initial model dump.')


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, previous_masks, dataset2idx, dataset2biases):
        self.args = args
        self.cuda = args.cuda
        self.model = model
        self.dataset2idx = dataset2idx
        self.dataset2biases = dataset2biases

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

            self.pruner = SparsePruner(
                self.model, self.args.prune_perc_per_layer, previous_masks,
                self.args.train_biases, self.args.train_bn)

    def eval(self, dataset_idx, biases=None):
        """Performs evaluation."""
        if not self.args.disable_pruning_mask:
            self.pruner.apply_mask(dataset_idx)
        if biases is not None:
            self.pruner.restore_biases(biases)

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
        if self.args.train_bn:
            self.model.train()
        else:
            self.model.train_nobn()
        return errors

    def do_batch(self, optimizer, batch, label):
        """Runs model for one batch."""
        if self.cuda:
            batch = batch.cuda()
            label = label.cuda()
        batch = Variable(batch)
        label = Variable(label)

        # Set grads to 0.
        self.model.zero_grad()

        # Do forward-backward.
        output = self.model(batch)
        self.criterion(output, label).backward()

        # Set fixed param grads to 0.
        if not self.args.disable_pruning_mask:
            self.pruner.make_grads_zero()

        # Update params.
        optimizer.step()

        # Set pruned weights to 0.
        if not self.args.disable_pruning_mask:
            self.pruner.make_pruned_zero()

    def do_epoch(self, epoch_idx, optimizer):
        """Trains model for one epoch."""
        for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
            self.do_batch(optimizer, batch, label)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        base_model = self.model

        # Prepare the ckpt.
        self.dataset2idx[self.args.dataset] = self.pruner.current_dataset_idx
        self.dataset2biases[self.args.dataset] = self.pruner.get_biases()
        ckpt = {
            'args': self.args,
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'dataset2idx': self.dataset2idx,
            'previous_masks': self.pruner.current_masks,
            'model': base_model,
        }
        if self.args.train_biases:
            ckpt['dataset2biases'] = self.dataset2biases

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
            if self.args.train_bn:
                self.model.train()
            else:
                self.model.train_nobn()
            self.do_epoch(epoch_idx, optimizer)
            errors = self.eval(self.pruner.current_dataset_idx)
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

        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)

    def prune(self):
        """Perform pruning."""
        print('Pre-prune eval:')
        self.eval(self.pruner.current_dataset_idx)

        self.pruner.prune()
        self.check(True)

        print('\nPost-prune eval:')
        errors = self.eval(self.pruner.current_dataset_idx)
        accuracy = 100 - errors[0]  # Top-1 accuracy.
        self.save_model(-1, accuracy, errors,
                        self.args.save_prefix + '_postprune')

        # Do final finetuning to improve results on pruned network.
        if self.args.post_prune_epochs:
            print('Doing some extra finetuning...')
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.args.lr, momentum=0.9,
                                  weight_decay=self.args.weight_decay)
            self.train(self.args.post_prune_epochs, optimizer, save=True,
                       savename=self.args.save_prefix + '_final', best_accuracy=accuracy)

        print('-' * 16)
        print('Pruning summary:')
        self.check(True)
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


def init_dump(arch):
    """Dumps pretrained model in required format."""
    if arch == 'vgg16':
        model = net.ModifiedVGG16()
    elif arch == 'vgg16bn':
        model = net.ModifiedVGG16BN()
    elif arch == 'resnet50':
        model = net.ModifiedResNet()
    elif arch == 'densenet121':
        model = net.ModifiedDenseNet()
    else:
        raise ValueError('Architecture type not supported.')

    previous_masks = {}
    for module_idx, module in enumerate(model.shared.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
            if 'cuda' in module.weight.data.type():
                mask = mask.cuda()
            previous_masks[module_idx] = mask
    torch.save({
        'dataset2idx': {'imagenet': 1},
        'previous_masks': previous_masks,
        'model': model,
    }, '../checkpoints/imagenet/%s.pt' % (arch))


def main():
    """Do stuff."""
    args = FLAGS.parse_args()
    if args.init_dump:
        init_dump(args.arch)
        return
    if args.prune_perc_per_layer <= 0:
        return

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
        previous_masks = []
    else:
        ckpt = torch.load(args.loadname)
        model = ckpt['model']
        previous_masks = ckpt['previous_masks']
        dataset2idx = ckpt['dataset2idx']
        if 'dataset2biases' in ckpt:
            dataset2biases = ckpt['dataset2biases']
        else:
            dataset2biases = {}

    # Add and set the model dataset.
    model.add_dataset(args.dataset, args.num_outputs)
    model.set_dataset(args.dataset)
    if args.cuda:
        model = model.cuda()

    # Create the manager object.
    manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases)

    # Perform necessary mode operations.
    if args.mode == 'finetune':
        # Make pruned params available for new dataset.
        manager.pruner.make_finetuning_mask()

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
    elif args.mode == 'prune':
        # Perform pruning.
        manager.prune()
    elif args.mode == 'check':
        # Load model and make sure everything is fine.
        manager.check(verbose=True)
    elif args.mode == 'eval':
        # Just run the model on the eval set.
        biases = None
        if 'dataset2biases' in ckpt:
            biases = ckpt['dataset2biases'][args.dataset]
        manager.eval(ckpt['dataset2idx'][args.dataset], biases)


if __name__ == '__main__':
    main()
