from Mat2Spec.data import *
from Mat2Spec.Mat2Spec import *
from Mat2Spec.file_setter import use_property
from Mat2Spec.utils import *
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import gc
import pickle
from copy import copy, deepcopy
import json
torch.autograd.set_detect_anomaly(True)
device = set_device()

# MOST CRUCIAL DATA PARAMETERS
parser = argparse.ArgumentParser(description='Mat2Spec')
parser.add_argument('--data_src', default='binned_dos_128',choices=['binned_dos_128','binned_dos_32','ph_dos_51'])
parser.add_argument('--label_scaling', default='standardized',choices=['standardized','normalized_sum', 'normalized_max'])
# MOST CRUCIAL MODEL PARAMETERS
parser.add_argument('--num_layers',default=3, type=int,
                    help='number of AGAT layers to use in model (default:3)')
parser.add_argument('--num_neurons',default=128, type=int,
                    help='number of neurons to use per AGAT Layer(default:128)')
parser.add_argument('--num_heads',default=4, type=int,
                    help='number of Attention-Heads to use  per AGAT Layer (default:4)')
parser.add_argument('--concat_comp',default=False, type=bool,
                    help='option to re-use vector of elemental composition after global summation of crystal feature.(default: False)')
parser.add_argument('--train_size',default=0.8, type=float, help='ratio size of the training-set (default:0.8)')
parser.add_argument('--trainset_subset_ratio',default=0.5, type=float, help='ratio size of the training-set subset (default:0.5)')
parser.add_argument('--use_catached_data', default=True, type=bool)
parser.add_argument("--train",action="store_true")  # default value is false
parser.add_argument('--num-epochs',default=200, type=int)
parser.add_argument('--batch-size',default=256, type=int)
parser.add_argument('--lr',default=0.001, type=float)
parser.add_argument('--Mat2Spec-input-dim',default=128, type=int)
parser.add_argument('--Mat2Spec-label-dim',default=128, type=int)
parser.add_argument('--Mat2Spec-latent-dim',default=128, type=int)
parser.add_argument('--Mat2Spec-emb-size',default=512, type=int)
parser.add_argument('--Mat2Spec-keep-prob',default=0.5, type=float)
parser.add_argument('--Mat2Spec-scale-coeff',default=1.0, type=float)
parser.add_argument('--Mat2Spec-loss-type',default='MAE', type=str, choices=['MAE', 'KL', 'WD', 'MSE'])
parser.add_argument('--Mat2Spec-K',default=10, type=int)
parser.add_argument("--finetune",action="store_true")  # default value is false
parser.add_argument("--ablation-LE",action="store_true")  # default value is false
parser.add_argument("--ablation-CL",action="store_true")  # default value is false
parser.add_argument("--finetune-dataset",default='null',type=str)
parser.add_argument('--check-point-path', default=None, type=str)
args = parser.parse_args(sys.argv[1:])

# GNN --- parameters
data_src = args.data_src
RSM = {'radius': 8, 'step': 0.2, 'max_num_nbr': 12}

number_layers                        = args.num_layers
number_neurons                       = args.num_neurons
n_heads                              = args.num_heads
concat_comp                          = args.concat_comp

# DATA PARAMETERS
random_num          =  1; random.seed(random_num)
np.random.seed(random_num)
torch.manual_seed(random_num)
# MODEL HYPER-PARAMETERS
num_epochs      = args.num_epochs
learning_rate   = args.lr
batch_size      = args.batch_size

stop_patience   = 150
best_epoch      = 1
adj_epochs      = 50
milestones      = [150,250]
train_param     = {'batch_size':batch_size, 'shuffle': True}
valid_param     = {'batch_size':batch_size, 'shuffle': False}

# DATALOADER/ TARGET NORMALIZATION
if args.data_src == 'binned_dos_128':
    pd_data = pd.read_csv(f'../Mat2Spec_DATA/label_edos/mpids.csv')
    np_data = np.load(f'../Mat2Spec_DATA/label_edos/total_dos_128.npy')
elif args.data_src == 'ph_dos_51':
    pd_data = pd.read_csv(f'../Mat2Spec_DATA/phdos/mpids.csv')
    np_data = np.load(f'../Mat2Spec_DATA/phdos/ph_dos.npy')
else:
    raise ValueError('')

NORMALIZER = DATA_normalizer(np_data)

CRYSTAL_DATA = CIF_Dataset(args, pd_data=pd_data, np_data=np_data, root_dir=f'../Mat2Spec_DATA/', **RSM)

if args.data_src == 'ph_dos_51':
    with open('../Mat2Spec_DATA/phdos/200801_trteva_indices.pkl', 'rb') as f:
        train_idx, val_idx, test_idx = pickle.load(f)
else:
    idx_list = list(range(len(pd_data)))
    random.shuffle(idx_list)
    train_idx_all, test_val = train_test_split(idx_list, train_size=args.train_size, random_state=random_num)
    test_idx, val_idx = train_test_split(test_val, test_size=0.5, random_state=random_num)

if args.trainset_subset_ratio < 1.0:
    train_idx, _ = train_test_split(train_idx_all, train_size=args.trainset_subset_ratio, random_state=random_num)
elif args.data_src != 'ph_dos_51':
    train_idx = train_idx_all

if args.finetune:
    assert args.finetune_dataset != 'null'
    if args.data_src == 'binned_dos_128':
        with open(f'../Mat2Spec_DATA/label_edos/materials_classes/' + args.finetune_dataset + '/train_idx.json', ) as f:
            train_idx = json.load(f)

        with open(f'../Mat2Spec_DATA/label_edos/materials_classes/' + args.finetune_dataset + '/val_idx.json', ) as f:
            val_idx = json.load(f)

        with open(f'../Mat2Spec_DATA/label_edos/materials_classes/' + args.finetune_dataset + '/test_idx.json', ) as f:
            test_idx = json.load(f)
    else:
        raise ValueError('Finetuning is only supported on the binned dos 128 dataset.')

#print('total size:', len(idx_list))
print('training size:', len(train_idx))
print('validation size:', len(val_idx))
print('testing size:', len(test_idx))
print('total size:', len(train_idx)+len(val_idx)+len(test_idx))

training_set       =  CIF_Lister(train_idx,CRYSTAL_DATA,df=pd_data)
validation_set     =  CIF_Lister(val_idx,CRYSTAL_DATA,df=pd_data)

print(f'> USING MODEL Mat2Spec!')
the_network = Mat2Spec(args, NORMALIZER)
net = the_network.to(device)

if args.finetune:
    # load checkpoint
    check_point = torch.load(args.check_point_path)
    net.load_state_dict(check_point['model'])
    learning_rate = learning_rate/5

# LOSS & OPTMIZER & SCHEDULER
optimizer = optim.AdamW(net.parameters(), lr = learning_rate, weight_decay = 1e-2)
#optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=0.9)

decay_times = 4
decay_ratios = 0.5
one_epoch_iter = np.ceil(len(train_idx) / batch_size)

if args.finetune:
    decay_ratios = 0.5

scheduler = lr_scheduler.StepLR(optimizer, one_epoch_iter * (num_epochs / decay_times), decay_ratios)

print(f'> TRAINING MODEL ...')
train_loader   = torch_DataLoader(dataset=training_set,   **train_param)
valid_loader   = torch_DataLoader(dataset=validation_set, **valid_param)

training_counter=0
training_loss=0
valid_counter=0
valid_loss=0
best_valid_loss=1e+10
check_fre = 10
current_step = 0

total_loss_smooth = 0
nll_loss_smooth = 0
nll_loss_x_smooth = 0
kl_loss_smooth = 0
cpc_loss_smooth = 0
prediction = []
prediction_x = []
label_gt = []
label_scale_value = []
sum_pred_smooth = 0

start_time = time.time()
for epoch in range(num_epochs):

    # TRAINING-STAGE
    net.train()
    args.train = True
    for data in tqdm(train_loader, mininterval=0.5, desc=f'(EPOCH:{epoch} TRAINING)', position=0, leave=True, ascii=True):
        current_step += 1
        data = data.to(device)
        train_label = deepcopy(data.y).to(device)
        if args.label_scaling == 'standardized':
            train_label_normalize = (train_label - NORMALIZER.mean.to(device)) / NORMALIZER.std.to(device)
        elif args.label_scaling == 'normalized_max':
            train_label_normalize = train_label / (torch.max(train_label,dim=1)[0].unsqueeze(1))
        elif args.label_scaling == 'normalized_sum':
            train_label_normalize = train_label / torch.sum(train_label, dim=1, keepdim=True)

        predictions = net(data)
        total_loss, nll_loss, nll_loss_x, kl_loss, c_loss, pred_e, pred_x = \
            compute_loss(train_label_normalize, predictions, NORMALIZER, args)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        prediction.append(pred_e.detach().cpu().numpy())
        prediction_x.append(pred_x.detach().cpu().numpy())
        label_gt.append(train_label.detach().cpu().numpy())

        total_loss_smooth += total_loss
        nll_loss_smooth += nll_loss
        nll_loss_x_smooth += nll_loss_x
        kl_loss_smooth += kl_loss
        cpc_loss_smooth += c_loss
        training_counter +=1

    total_loss_smooth = total_loss_smooth / training_counter
    nll_loss_smooth = nll_loss_smooth / training_counter
    nll_loss_x_smooth = nll_loss_x_smooth / training_counter
    kl_loss_smooth = kl_loss_smooth / training_counter
    cpc_loss_smooth = cpc_loss_smooth / training_counter

    prediction = np.concatenate(prediction, axis=0)
    prediction_x = np.concatenate(prediction_x, axis=0)
    label_gt = np.concatenate(label_gt, axis=0)

    if args.label_scaling == 'standardized':
        mean = NORMALIZER.mean.detach().numpy()
        std = NORMALIZER.std.detach().numpy()
        label_gt_standardized = (label_gt-mean)/std
        mae = np.mean(np.abs((prediction)-label_gt_standardized))
        mae_x = np.mean(np.abs((prediction_x)-label_gt_standardized))
        prediction = prediction*std+mean
        prediction_x = prediction_x*std+mean
        prediction[prediction < 0] = 0
        prediction_x[prediction_x < 0] = 0
        mae_ori = np.mean(np.abs((prediction)-label_gt))
        mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))

    elif args.label_scaling == 'normalized_max':
        label_max = np.expand_dims(np.max(label_gt, axis=1), axis=1)
        label_gt_standardized = label_gt / label_max
        mae = np.mean(np.abs((prediction)-label_gt_standardized))
        mae_x = np.mean(np.abs((prediction_x)-label_gt_standardized))
        prediction = prediction*label_max
        prediction_x = prediction_x*label_max
        mae_ori = np.mean(np.abs((prediction)-label_gt))
        mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))

    elif args.label_scaling == 'normalized_sum':
        assert args.Mat2Spec_loss_type == 'KL' or args.Mat2Spec_loss_type == 'WD'
        label_sum = np.sum(label_gt, axis=1, keepdims=True)
        label_gt_standardized = label_gt / label_sum
        mae = np.mean(np.abs((prediction)-label_gt_standardized))
        mae_x = np.mean(np.abs((prediction_x)-label_gt_standardized))
        prediction = prediction*label_sum
        prediction_x = prediction_x*label_sum
        mae_ori = np.mean(np.abs((prediction)-label_gt))
        mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))

    print("\n********** TRAINING STATISTIC ***********")
    print("total_loss =%.6f\t nll_loss =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t cpc_loss=%.6f\t" %
          (total_loss_smooth, nll_loss_smooth, nll_loss_x_smooth, kl_loss_smooth, cpc_loss_smooth))
    print("mae=%.6f\t mae_x=%.6f\t mae_ori=%.6f\t mae_x_ori=%.6f" % (mae, mae_x, mae_ori, mae_x_ori))
    print("\n*****************************************")

    training_counter = 0
    total_loss_smooth = 0
    nll_loss_smooth = 0
    nll_loss_x_smooth = 0
    kl_loss_smooth = 0
    cpc_loss_smooth = 0
    prediction = []
    prediction_x = []
    label_gt = []
    label_scale_value = []
    sum_pred_smooth = 0

    # VALIDATION-PHASE
    net.eval()
    for data in tqdm(valid_loader, mininterval=0.5, desc='(validating)', position=0, leave=True, ascii=True):
        data = data.to(device)
        valid_label = deepcopy(data.y).float().to(device)

        if args.label_scaling == 'standardized':
            valid_label_normalize = (valid_label - NORMALIZER.mean.to(device)) / NORMALIZER.std.to(device)
        elif args.label_scaling == 'normalized_max':
            valid_label_normalize = valid_label/(torch.max(valid_label,dim=1)[0].unsqueeze(1))

        elif args.label_scaling == 'normalized_sum':
            valid_label_normalize = valid_label / (torch.sum(valid_label, dim=1, keepdim=True)+1e-8)

        with torch.no_grad():
            predictions = net(data)
            total_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, pred_e, pred_x = \
                compute_loss(valid_label_normalize, predictions, NORMALIZER, args)

        prediction.append(pred_e.detach().cpu().numpy())
        prediction_x.append(pred_x.detach().cpu().numpy())
        label_gt.append(valid_label.detach().cpu().numpy())

        total_loss_smooth += total_loss
        nll_loss_smooth += nll_loss
        nll_loss_x_smooth += nll_loss_x
        kl_loss_smooth += kl_loss
        cpc_loss_smooth += cpc_loss
        valid_counter += 1

    total_loss_smooth = total_loss_smooth / valid_counter
    nll_loss_smooth = nll_loss_smooth / valid_counter
    nll_loss_x_smooth = nll_loss_x_smooth / valid_counter
    kl_loss_smooth = kl_loss_smooth / valid_counter
    cpc_loss_smooth = cpc_loss_smooth / valid_counter

    prediction = np.concatenate(prediction, axis=0)
    prediction_x = np.concatenate(prediction_x, axis=0)
    label_gt = np.concatenate(label_gt, axis=0)

    if args.label_scaling == 'standardized':
        mean = NORMALIZER.mean.detach().numpy()
        std = NORMALIZER.std.detach().numpy()
        label_gt_standardized = (label_gt-mean)/std
        mae = np.mean(np.abs((prediction)-label_gt_standardized))
        mae_x = np.mean(np.abs((prediction_x)-label_gt_standardized))
        prediction = prediction*std+mean
        prediction_x = prediction_x*std+mean
        prediction[prediction < 0] = 0
        prediction_x[prediction_x < 0] = 0
        mae_ori = np.mean(np.abs((prediction)-label_gt))
        mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))

    elif args.label_scaling == 'normalized_max':
        label_max = np.expand_dims(np.max(label_gt, axis=1), axis=1)
        label_gt_standardized = label_gt / label_max
        mae = np.mean(np.abs((prediction)-label_gt_standardized))
        mae_x = np.mean(np.abs((prediction_x)-label_gt_standardized))
        prediction = prediction*label_max
        prediction_x = prediction_x*label_max
        mae_ori = np.mean(np.abs((prediction)-label_gt))
        mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))

    elif args.label_scaling == 'normalized_sum':
        assert args.Mat2Spec_loss_type == 'KL' or args.Mat2Spec_loss_type == 'WD'
        label_sum = np.sum(label_gt, axis=1, keepdims=True)
        label_gt_standardized = label_gt / label_sum
        mae = np.mean(np.abs((prediction)-label_gt_standardized))
        mae_x = np.mean(np.abs((prediction_x)-label_gt_standardized))
        prediction = prediction*label_sum
        prediction_x = prediction_x*label_sum
        mae_ori = np.mean(np.abs((prediction)-label_gt))
        mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))

    print("\n********** VALIDATING STATISTIC ***********")
    print("total_loss =%.6f\t nll_loss =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t cpc_loss = %.6f\t" %
          (total_loss_smooth, nll_loss_smooth, nll_loss_x_smooth, kl_loss_smooth, cpc_loss_smooth))
    print("mae=%.6f\t mae_x=%.6f\t mae_ori=%.6f\t mae_x_ori=%.6f" % (mae, mae_x, mae_ori, mae_x_ori))
    print("\n*****************************************")

    if best_valid_loss > mae_x_ori:
        best_valid_loss = mae_x_ori
        print("\n********** SAVING MODEL ***********")
        checkpoint = {'model': net.state_dict(), 'args': args}
        if not args.finetune:
            #checkpoint_path = './TRAINED/'
            save_path = './TRAINED/model_Mat2Spec_' + args.data_src + '_' + args.label_scaling + '_' \
                        + args.Mat2Spec_loss_type + '_trainsize' + str(args.trainset_subset_ratio)
        else:
            save_path = './TRAINED/finetune/model_Mat2Spec_' + args.data_src + '_' + args.label_scaling + '_' \
                        + args.Mat2Spec_loss_type + '_finetune_' + str(args.finetune_dataset)

        if args.ablation_LE:
            save_path = save_path + '_ablation_LE'

        if args.ablation_CL:
            save_path = save_path + '_ablation_CL'

        save_path = save_path + '.chkpt'
        torch.save(checkpoint, save_path)
        print("A new model has been saved to " + save_path)
        print("\n*****************************************")

    valid_counter=0
    total_loss_smooth = 0
    nll_loss_smooth = 0
    nll_loss_x_smooth = 0
    kl_loss_smooth = 0
    cpc_loss_smooth = 0
    prediction = []
    prediction_x = []
    label_gt = []
    label_scale_value = []
    sum_pred_smooth = 0
    gc.collect()

end_time = time.time()
e_time = end_time - start_time
print('Best validation loss=%.6f, training time (min)=%.6f'%(best_valid_loss, e_time/60))
print(f"> DONE TRAINING !")
