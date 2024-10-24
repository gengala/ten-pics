from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import functools
import argparse
import torch
import git
import os

print = functools.partial(print, flush=True)

from trainers import training_pic, test_pc
from pic import PIC, zw_quadrature, pc2integral_group_args
from utils import init_random_seeds, get_date_time_str, count_trainable_parameters, param_to_buffer, freeze_mixing_layers
from data import datasets


from tenpcs.models.functional import integrate
from tenpcs.layers.input.categorical import CategoricalLayer
from tenpcs.layers.sum_product import CollapsedCPLayer, TuckerLayer
from tenpcs.region_graph import QuadTree, QuadGraph
from tenpcs.models import TensorizedPC


parser = argparse.ArgumentParser()
parser.add_argument("--seed",            type=int,   default=42,         help="random seed")
parser.add_argument("--gpu",             type=int,   default=0,          help="device on which run the experiment")
parser.add_argument("--root",            type=str,   default='./data/',  help="root dataset dir")
parser.add_argument("--dataset", "-ds",  type=str,   default="mnist",    help="dataset for the experiment")
parser.add_argument("--split",           type=str,   default=None,       help='dataset split for EMNIST')
parser.add_argument("--out-dir",         type=str,   default="out/pic",  help="output dir for saving logs and models")
parser.add_argument("--batch-size",      type=int,   default=256,        help="batch size")
parser.add_argument("--num-workers",     type=int,   default=8,          help="data loader num workers")
parser.add_argument("--lr",              type=float, default=0.005,      help="learning rate")
parser.add_argument("--weight-decay",    type=float, default=1e-2,       help="weight decay coefficient")
parser.add_argument("--t0",              type=int,   default=500,        help='scheduler CAWR t0, 1 for fixed lr ')
parser.add_argument("--eta-min",         type=float, default=1e-4,       help='scheduler CAWR eta min')
parser.add_argument("--loss-reduction",  type=str,   default="sum",      help="loss reduction: 'mean', 'sum'")
parser.add_argument("--max-num-epochs",  type=int,   default=200,        help="max num epochs")
parser.add_argument('--valid-freq',      type=int,   default=250,        help='validation step every valid-freq steps')
parser.add_argument("--patience",        type=int,   default=5,          help='patience for early stopping')
parser.add_argument("--min-delta",       type=float, default=0,          help='min delta early stopping')
parser.add_argument("--k",               type=int,   default=128,        help="num integration points")
parser.add_argument("--a",               type=float, default=-1,         help="inf support")
parser.add_argument("--b",               type=float, default=1,          help="sup support")
parser.add_argument("--rg",              type=str,   default="QT",       help="region graph: 'BT', 'QG', 'QT'")
parser.add_argument("--inner-layer",     type=str,   default="cp",       help="inner layer type: 'tucker' or 'cp'")
parser.add_argument("--net-dim",         type=int,   default=256,        help="pic neural net dim")
parser.add_argument("--input-sharing",   type=str,   default="f",        help="input sharing: either 'none', 'c', 'f'")
parser.add_argument("--inner-sharing",   type=str,   default="c",        help="input sharing: either 'none', 'c', 'f'")
parser.add_argument("--ff-dim",          type=int,   default=None,       help="fourier features output dim")
parser.add_argument("--sigma",           type=float, default=1.0,        help="sigma fourier features")
parser.add_argument("--ycc",             type=str,   default="none",     help="either 'none', 'lossless', 'lossy'")
parser.set_defaults(learn_ff=False)
parser.add_argument('-ff',         dest='learn_ff',            action='store_true')
parser.add_argument('-no-ff',      dest='learn_ff',            action='store_false')
parser.set_defaults(bias=True)
parser.add_argument('-bias',       dest='bias',                action='store_true')
parser.add_argument('-no-bias',    dest='bias',                action='store_false')
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

num_categories = 256
num_channels = train[0].size(-1)
if args.input_sharing == "f":
    CategoricalLayer.full_sharing = True
qpc = TensorizedPC.from_region_graph(
    rg=rg,
    layer_cls=INNER_LAYERS[args.inner_layer],
    efamily_cls=CategoricalLayer,
    efamily_kwargs={'num_categories': num_categories},
    num_inner_units=args.k,
    num_input_units=args.k,
    num_channels=train[0].size(-1),
)
freeze_mixing_layers(qpc)
num_qpc_params = count_trainable_parameters(qpc)  # this must be here, because params are going to buffer next
param_to_buffer(qpc)

pic = PIC(
    integral_group_args=pc2integral_group_args(qpc),
    num_vars=qpc.num_vars,
    input_layer_type='categorical',
    num_categories=num_categories * num_channels,
    net_dim=args.net_dim,
    bias=args.bias,
    input_sharing=args.input_sharing,
    inner_sharing=args.inner_sharing,
    ff_dim=args.ff_dim,
    sigma=args.sigma,
    learn_ff=args.learn_ff
).to(device)
print(pic)

print(f"QPC num of params: {num_qpc_params}")
print(f"PIC num of params: {count_trainable_parameters(pic)}")

###############################################################################
################################ training loop ################################
###############################################################################

optimizer = torch.optim.Adam(pic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1, eta_min=args.eta_min)
z_quad, w_quad = zw_quadrature('trapezoidal', nip=args.k, a=args.a, b=args.b, device=device)

training_pic(
    pic=pic,
    qpc=qpc,
    z_quad=z_quad,
    w_quad=w_quad,
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

###################################################################################
################################ testing & logging ################################
###################################################################################

pic = torch.load(model_dir).eval()
pic.parameterize_qpc(qpc=qpc, z_quad=z_quad, w_quad=w_quad)
print('norm const', integrate(qpc)(None).item())
print(dataset_str)
print('(PIC) %s-%s-%d-%s' % (args.rg, args.inner_layer, args.k, args.ycc))
results = test_pc(qpc, train_loader, valid_loader, test_loader)

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
        'inner_layer':      list(INNER_LAYERS.keys())
    },
)
writer.close()
