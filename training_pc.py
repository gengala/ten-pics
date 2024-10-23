from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import functools
import numpy as np
import argparse
import torch
import git
import os

print = functools.partial(print, flush=True)


from trainers import training_pc, test_pc
from utils import init_random_seeds, get_date_time_str, param_to_buffer, count_pc_params, count_trainable_parameters, freeze_mixing_layers
from data import datasets
from tensorized_circuit import TensorizedPC


# cirkit
from tenpcs.layers.input.exp_family.categorical import CategoricalLayer
from tenpcs.layers.input.exp_family.normal import NormalLayer
from tenpcs.layers.sum_product import CollapsedCPLayer, TuckerLayer
from tenpcs.region_graph import QuadTree, QuadGraph


parser = argparse.ArgumentParser()
parser.add_argument("--seed",               type=int,   default=42,         help="random seed")
parser.add_argument("--gpu",                type=int,   default=0,          help="device on which run the experiment")
parser.add_argument("--root",               type=str,   default='./data/',  help="root dataset dir")
parser.add_argument("--dataset", "-ds",     type=str,   default="mnist",    help="dataset for the experiment")
parser.add_argument("--split",              type=str,   default=None,       help='dataset split for EMNIST')
parser.add_argument("--out-dir",            type=str,   default="out/pc",   help="output dir for saving logs and models")
parser.add_argument("--batch-size",         type=int,   default=256,        help="batch size")
parser.add_argument("--num-workers",        type=int,   default=8,          help="data loader num workers")
parser.add_argument("--lr",                 type=float, default=0.01,       help="learning rate")
parser.add_argument("--weight-decay",       type=float, default=0,          help="weight decay coefficient")
parser.add_argument("--t0",                 type=int,   default=1,          help='scheduler CAWR t0, 1 for fixed lr ')
parser.add_argument("--eta-min",            type=float, default=1e-4,       help='scheduler CAWR eta min')
parser.add_argument("--loss-reduction",     type=str,   default="sum",      help="loss reduction: 'mean', 'sum'")
parser.add_argument("--max-num-epochs",     type=int,   default=200,        help="max num epochs")
parser.add_argument('--valid-freq',         type=int,   default=250,        help='validation every n steps')
parser.add_argument("--patience",           type=int,   default=5,          help='patience for early stopping')
parser.add_argument("--min-delta",          type=float, default=0,          help='min delta early stopping')
parser.add_argument("--k",                  type=int,   default=128,        help="num categories for mixtures")
parser.add_argument("--rg",                 type=str,   default="QT",       help="region graph: 'PD', 'QG', 'QT'")
parser.add_argument("--inner-layer",        type=str,   default="cp",       help="inner layer type: 'tucker' or 'cp'")
parser.add_argument("--input-layer",        type=str,   default="cat",      help="input type: either 'cat' or 'bin'")
parser.add_argument("--ycc",                type=str,   default="none",     help="either 'none', 'lossless', 'lossy'")
parser.set_defaults(freeze_mixing_layers=True)
parser.add_argument('-fml',        dest='freeze_mixing_layers',   action='store_true')
parser.add_argument('-no-fml',     dest='freeze_mixing_layers',   action='store_false')
parser.set_defaults(shared_input_layer=False)
parser.add_argument('-sil',        dest='shared_input_layer',   action='store_true',        help='multi_head')
parser.add_argument('-no-sil',     dest='shared_input_layer',   action='store_false',       help='multi_head')
args = parser.parse_args()
init_random_seeds(seed=args.seed)

print('\n\n\n')
args.git_commit = git.Repo(search_parent_directories=True).head.object.hexsha
args.time_stamp = time_stamp = get_date_time_str()
for key, value in vars(args).items():
    print(f"{key}: {value}")
print('\n')

dataset_str = args.dataset + ('' if args.split is None else ('_' + args.split))
INNER_LAYERS = {"tucker": TuckerLayer, "cp": CollapsedCPLayer}
INPUT_LAYERS = {"cat": CategoricalLayer, "normal": NormalLayer}
device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"

##########################################################################
################### load dataset & create logging dirs ###################
##########################################################################

train, valid, test = datasets.load_dataset(args.dataset, split=args.split, root=args.root, valid_split_percentage=0.05, ycc=args.ycc)
print('train-valid-test lengths:', len(train), len(valid), len(test))

train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

model_dir = os.path.join(args.out_dir, "models", dataset_str, f"{args.rg}_{args.inner_layer}_{args.k}", time_stamp + ".pt")
log_dir = os.path.join(args.out_dir, "logs", dataset_str, f"{args.rg}_{args.inner_layer}_{args.k}", time_stamp)
if not os.path.exists(os.path.dirname(model_dir)): os.makedirs(os.path.dirname(model_dir))
writer = SummaryWriter(log_dir=log_dir)

#######################################################################################
################################## instantiate model ##################################
#######################################################################################

if args.rg == 'QG':
    image_size = int(np.sqrt(train[0].shape[0]))
    rg = QuadGraph(width=image_size, height=image_size)
elif args.rg == 'QT':
    image_size = int(np.sqrt(train[0].shape[0]))
    rg = QuadTree(width=image_size, height=image_size)
else:
    raise NotImplementedError("region graph not available")


pc = TensorizedPC.from_region_graph(
    rg=rg,
    layer_cls=INNER_LAYERS[args.inner_layer],
    efamily_cls=INPUT_LAYERS[args.input_layer],
    efamily_kwargs={'num_categories': 256},
    num_inner_units=args.k,
    num_input_units=args.k,
    num_channels=train[0].size(-1)
)
if args.freeze_mixing_layers:
    freeze_mixing_layers(pc)
if args.shared_input_layer:
    input_layer = torch.nn.Parameter(pc.input_layer.params.param[:1])
    param_to_buffer(pc.input_layer)
else:
    input_layer = None


print(pc)
print(f"Num params PC: {count_pc_params(pc)}")
print(f"Num trainable params: {count_trainable_parameters(pc) + (0 if input_layer is None else input_layer.nelement())}")

###############################################################################
################################ training loop ################################
###############################################################################

optimizer = torch.optim.Adam([
    {'params': input_layer if args.shared_input_layer else pc.input_layer.parameters()},
    {'params': pc.inner_layers.parameters(), 'weight_decay': args.weight_decay}], lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1, eta_min=args.eta_min)

training_pc(
    pc=pc,
    input_layer=input_layer,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_reduction=args.loss_reduction,
    max_train_steps=int(len(train) // args.batch_size * args.max_num_epochs),
    patience=args.patience,
    min_delta=args.min_delta,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    valid_freq=args.valid_freq,
    writer=writer,
    model_dir=model_dir
)

#########################################################################
################################ testing ################################
#########################################################################

pc: TensorizedPC = torch.load(model_dir)
print(dataset_str)
print('(PC) %s-%s-%d-%s' % (args.rg, args.inner_layer, args.k, args.ycc))
results = test_pc(pc, train_loader, valid_loader, test_loader)

writer.add_hparams(
    hparam_dict=vars(args),
    metric_dict={
        'best/train/ll':    results['train_ll'],
        'best/train/bpd':   results['train_bpd'],
        'best/valid/ll':    results['valid_ll'],
        'best/valid/bpd':   results['valid_bpd'],
        'best/test/ll':     results['test_ll'],
        'best/test/bpd':    results['test_bpd'],
    },
    hparam_domain_discrete={
        'rg':               ['QG', 'QT'],
        'inner_layer':      list(INNER_LAYERS.keys()),
        'input_layer':      list(INPUT_LAYERS.keys())
    },
)
writer.close()
