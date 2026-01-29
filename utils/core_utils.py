import numpy as np
import torch
from utils.utils import *
from data_augmentation_method.wsi_feature_augmentation import *
import os
import re
from models.RDT import RDT
from models.LFFA import LFFA,WSIDataset,collate_wsi_batch,info_nce_logits_targets,compute_grad_match_loss,compute_infoNCE_loss
from models.FSCIL_aug import FSCIL
from models.FATL import FATL
import random
from utils.fm_set import get_augmentation_path
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from dataset_modules.dataset_generic import Generic_MIL_Dataset, save_splits, Generic_WSI_Dataset
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_transmil import TransMIL
from models.model_armil import ARMIL, ARMIL_atten, get_relation
from models.model_abmil import ABMIL
from models.model_max_abmil import MABMIL
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc
from tqdm import tqdm
from models.model_AR_ssl import ARSSL as nodelete_ARSSL

from collections import OrderedDict
from topk.svm import SmoothTop1SVM
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report,balanced_accuracy_score, f1_score
import time
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import matplotlib
from matplotlib.colors import ListedColormap
#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.linear import LinearProbe, weight_sampler_LinearProbe,MultiHeadLinear,FFPM,FFCA

    
class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        if torch.is_tensor(Y_hat):

            Y_hat = Y_hat.detach().view(-1).cpu()   # [B]
            Y = Y.detach().view(-1).cpu()           # [B]

            for y_hat_i, y_i in zip(Y_hat, Y):
                y_hat_i = int(y_hat_i.item())
                y_i = int(y_i.item())

                self.data[y_i]["count"] += 1
                self.data[y_i]["correct"] += (y_hat_i == y_i)

        else:
            Y_hat = int(Y_hat)
            Y = int(Y)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat == Y)

    def log_wsi(self, Y_hat, Y):
        for label_class in torch.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

         
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=10, activation=nn.ReLU, dropout=0.0):

        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h


        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if val_split is not None:
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    else:
        save_splits(datasets, ['train', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    if val_split is not None:
        print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    args.max_window_size=[7]
    if args.model_type in ['linear','finetuning','random_init']:
        if args.aug_method == 'LFFA':

            if args.ablation == 5:
                from models.LFFA_FM_learn import LFFA as LFFA_linear
                model = LFFA_linear(input_dim=args.embed_dim, mhsa_heads = 8,mlp_mult = 4, depth = 1, 
                                mlp_activation = "geglu",num_classifier = args.n_classes,
                                aug_para = args.aug_para, aug_method = args.aug_method_LFFA, class_way = args.head_way
                                )
            elif args.ablation<5:
                from models.LFFA_FM_learn_ablation import LFFA as LFFA_linear
                model = LFFA_linear(input_dim=args.embed_dim, mhsa_heads = 8,mlp_mult = 4, depth = 1, 
                                mlp_activation = "geglu",num_classifier = args.n_classes,
                                aug_para = args.aug_para, aug_method = args.aug_method_LFFA, class_way = args.head_way, ablation= args.ablation)
    
    elif args.model_type in ['mlp']:
        model = MLP(input_dim=args.embed_dim, hidden_dims=[256, 128], output_dim=args.n_classes, activation=nn.ReLU, dropout=0.1)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    if args.model_name in ['CHIEF','PRISM','TITAN','TANGLE','FEATHER','GIGAPATH']:
        model_base = None
    else:
        raise NotImplementedError('Model "{}" not supported'.format(args.model_name))

    if args.multi_gpu == 0 or len(args.multi_gpu) == 1:
        torch.cuda.set_device(args.multi_gpu[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ = model.to(device)
        if args.model_name is not None and model_base is not None:
            model_base = model_base.to(device)
    else:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(f"cuda:{args.multi_gpu[0]}")
        model = nn.DataParallel(model, device_ids=args.multi_gpu) 
        #model = model.to(device)
        if args.model_name is not None and model_base is not None:
            model_base = model_base.to(device)
            model_base1 = model_base1.to(device)
        model = model.to(device)
    print('Done!')
    if args.model_name in ['TITAN_un']:
        print_network(model)

    print('\nInit optimizer ...', end=' ')
    if args.model_type in ['finetuning','random_init'] and model_base is not None:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(model_base.parameters()), lr=args.lr, weight_decay=args.reg)
    else:
        optimizer = get_optim(model, args)
    print('Done!')
    print('\nInit Loaders...', end=' ')
    if issubclass(train_split.__class__, Generic_MIL_Dataset):
        if args.model_type in ['lwmil123']:
            train_split.load_from_h5(True)
            train_split.return_slideid()
            if val_split is not None:
                val_split.load_from_h5(True)
                val_split.return_slideid() 
                val_loader = get_caption_image_loader(val_split,training = True, weighted = False, batch_size=1,num_workers=1)
            else:
                val_loader = None
            test_split.load_from_h5(True)
            test_split.return_slideid()
            train_loader = get_caption_image_loader(train_split,training = True, weighted = True, batch_size=1,num_workers=1)
            
            test_loader = get_caption_image_loader(test_split,training = False, weighted = False, batch_size=1,num_workers=1)
        else:
            train_split.load_from_h5(True)
            train_split.return_slideid()
            test_split.load_from_h5(True)
            test_split.return_slideid()
            if val_split is not None:
                val_split.load_from_h5(True)
                val_split.return_slideid() 
                val_loader = get_coords_id_loader(val_split,training = True, weighted = False, batch_size=1,num_workers=1)
            else:
                val_loader = None
            if args.model_type in ['logistic_regression','KNN','tsne']:
                train_loader = get_coords_id_loader(train_split,training = True, weighted = True, batch_size=1,num_workers=1)
            else:
                train_loader = get_coords_id_loader(train_split,training = True, weighted = True, batch_size=1,num_workers=1)
            test_loader = get_coords_id_loader(test_split,training = False, weighted = False, batch_size=1,num_workers=1)
    elif issubclass(train_split.__class__, Generic_WSI_Dataset):
        train_split.load_from_h5(True)
        train_split.return_slideid() 
        test_split.load_from_h5(True)
        test_split.return_slideid() 
        if val_split is not None:
            val_split.load_from_h5(True)
            val_split.return_slideid() 
            val_loader = get_wsi_loader(val_split,training = True, weighted = False, batch_size=1,num_workers=1)
        else:
            val_loader = None
        if args.model_type in ['logistic_regression','KNN','tsne','linear']:
            train_loader = get_wsi_loader(train_split,training = False, weighted = False, batch_size=1,num_workers=1)
        elif args.model_type in ['mlp'] and args.aug_method not in ['data_weight_sampler']:
            train_loader = get_wsi_loader(train_split,training = False, weighted = False, batch_size=1,num_workers=1)
        else:
            train_loader = get_wsi_loader(train_split,training = True, weighted = True, batch_size=1,num_workers=1)
        test_loader = get_wsi_loader(test_split,training = False, weighted = False, batch_size=1,num_workers=1)
    else:
        raise NotImplementedError('Dataset type "{}" not supported'.format(train_split.__class__))


    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 10, stop_epoch=20, verbose = True)#(patience = 13, stop_epoch=20, verbose = True)'
        #early_stopping = EarlyStopping(patience = 1, stop_epoch= 3, verbose = True)
    else:
        early_stopping = None
    print('Done!')
    
    if args.model_type in ['linear','finetuning','random_init','mlp']:
        aug_train_loader = get_augmentation_fea(train_loader,val_loader,
                        test_loader,args,cur,early_stopping,device)
        #train_features, train_labels, val_features, val_labels,test_features, test_labels,val_slide_ids,test_slide_ids = get_wsi_loader_fea(train_loader,val_loader,test_loader,args,cur)
    for epoch in (range(args.max_epochs)):

        if args.model_type in ['linear','finetuning','random_init','mlp']:
            train_linear_loop(epoch, model, aug_train_loader, optimizer, args.n_classes, writer, loss_fn,model_base = model_base,
                       num_region = args.num_region, args=args)
            stop = validate_linear(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir,model_base = model_base,num_region = args.num_region, args=args)
        else:
            #print(epoch,2)
            stop = True
        if stop: 
            break
    
    
    if True:
        if args.early_stopping:
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
        else:
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
        if args.model_type in ['mlp']: 
            _, val_error, val_auc, _ = validate_linear(cur, epoch, model, val_features,val_labels,val_slide_ids, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir,model_base = model_base,num_region = args.num_region, args=args, return_auc = True)
        else:
            _, val_error, val_auc, _= summary(model, val_loader, args.n_classes,args.model_type, max_window_size=args.max_window_size, 
                                        patch_size =args.step_size,model_base = model_base,num_region = args.num_region, args=args)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
        if args.model_type in ['mlp']: 
            results_dict, test_error, test_auc, acc_logger = validate_linear(cur, epoch, model, test_features,test_labels, test_slide_ids, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir,model_base = model_base,num_region = args.num_region, args=args,return_auc = True)
        else: 
            results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes,args.model_type, max_window_size=args.max_window_size, 
                                                                patch_size =args.step_size,model_base = model_base,num_region = args.num_region, args=args
                                                                )
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_auc', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
            writer.close()
        return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 
    









    
def make_weights_for_balanced_classes_split_noaug(dataset,counts):
	N = float(len(dataset))        
	#print(counts)                                   
	weight_per_class = [N/(counts[c]) for c in range(len(counts))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)
def get_augmentation_fea(train_loader,val_loader,test_loader,args,cur,early_stopping,device):
    features=[]
    labels=[]
    train_slide_ids = []
    for data, label, slide_id in train_loader:
        features.append(data)
        labels.append(label.numpy())
        train_slide_ids.extend(slide_id)

    patch_train_fea = features
    #train_features = torch.cat([fea[:, 0, :] for fea in features], dim=0)
    train_labels=np.concatenate(labels)
    
    slide_data = pd.read_csv(args.csv_path)
    aug_slide = pd.read_csv(os.path.join(args.split_aug_path,'augmentaion_num_fold_{}.csv'.format(cur)))
    slide_data['wsi_ID']   = slide_data['wsi_ID'].astype(str)
    slide_data['slide_id'] = slide_data['slide_id'].astype(str)
    aug_slide['slide_id']  = aug_slide['slide_id'].astype(str)
    aug_slide['expand_count'] = pd.to_numeric(aug_slide['expand_count'], errors='coerce').fillna(0)
    wsi2slide = (slide_data
            .drop_duplicates('wsi_ID', keep='last')
            .assign(wsi_ID=lambda df: df['wsi_ID'].str.replace(args.data_formats, '', regex=True))
            .set_index('wsi_ID')['slide_id']
            .to_dict())
    slide2exp = aug_slide.set_index('slide_id')['expand_count'].to_dict()
    exp_list = []
    for idx, wsi in enumerate(train_slide_ids):   
        sid = wsi2slide.get(wsi)          
        if sid is None:                   
            print(args.csv_path)
            raise NotImplementedError('File "{}" is error'.format(args.csv_path))
        exp = slide2exp.get(sid, 0)      
        exp_list.append(exp)
    num_aug = torch.tensor(exp_list, dtype=torch.long)

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    if 'shot' in args.aug_task:
        keep_idx = (num_aug != -1).nonzero(as_tuple=True)[0]
        patch_train_fea = [patch_train_fea[i] for i in keep_idx.tolist()]
        train_labels = [train_labels[i] for i in keep_idx.tolist()]
        train_slide_ids = [train_slide_ids[i] for i in keep_idx.tolist()]
        mask = num_aug != -1        # bool tensor, shape [520]
        num_aug_new = num_aug[mask]
        m_aug = re.match(r'(\d+)-shot_aug_(\d+)', args.aug_task)
        base_shot = int(m_aug.group(1))  # 2
        target = int(m_aug.group(2))     # 4
        if target % base_shot != 0:
            raise ValueError(f"Invalid aug_task {args.aug_task}: target ({target}) "
                            f"must be a multiple of base_shot ({base_shot}).")

        dup_factor = target // base_shot     
        aug_times_per_sample = dup_factor - 1 
        num_aug_new = (torch.ones(len(train_labels),dtype=torch.long)*aug_times_per_sample)
    else:
        num_aug_new = num_aug
    if 'shot' in args.aug_task and args.aug_method in ['brightness','contrast',]:
        aug_feas = []
        for idx, slide_id in enumerate(train_slide_ids):
            if args.aug_method in ['contrast']:
                aug_pixel_fea_path = os.path.join('fm_aug_fea/wsi_embedding/',args.model_name.lower(),'%s.npy'%slide_id)
            elif args.aug_method in ['brightness']:
                aug_pixel_fea_path = os.path.join('fm_aug_fea_brightness/wsi_embedding/',args.model_name.lower(),'%s.npy'%slide_id)
            aug_feature = np.load(aug_pixel_fea_path, allow_pickle=True)
            aug_feature = torch.from_numpy(aug_feature)
            patch_train_fea.append(aug_feature.unsqueeze(0))
            train_labels.append(train_labels[idx])
        num_aug_new = (torch.ones(len(train_labels),dtype=torch.long)*aug_times_per_sample)
        

    dataset = WSIDataset(patch_train_fea, train_labels,num_aug_new)
    if args.aug_task == 'no_aug':
        counts = np.bincount(train_labels)
        weights = make_weights_for_balanced_classes_split_noaug(dataset,counts)#_noaug
        train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=2,
                sampler = WeightedRandomSampler(weights, len(weights)),
                collate_fn=collate_wsi_batch,
                generator=g,
            )
    else:
        train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=2,
                shuffle=True,
                collate_fn=collate_wsi_batch,
                generator=g,
            )
    return train_loader







def train_linear_loop(epoch, model, aug_train_loader, optimizer, n_classes, writer = None, loss_fn = None,model_base = None,num_region = None,args = None):   
    model.train()
    
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    #print(n_classes)
    train_loss = 0.
    train_error = 0.
    train_sum = 0.
    from torch.amp import autocast, GradScaler
    soft_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.aug_para3, reduction='mean')

    print('\n')

    for data, target, labels, attn_mask,aug_exp in aug_train_loader:
        # data:      [B, Max_L, 512]
        # target:    [B, 512]
        # labels:    [B]
        # attn_mask: [B, Max_L] (bool)
        # aug_exp: [B]
        B = data.shape[0]
        data = data.to(device)
        target = target.to(device)
        labels = labels.to(device)
        aug_exp = aug_exp.to(device)
        attn_mask = attn_mask.to(device)
        optimizer.zero_grad()
        
        all_logits, aug_labels= model(data,target,labels,aug_exp, attn_mask)
        B_aug = all_logits.shape[0]
        logits = all_logits[:B]
        Y_hat = torch.argmax(logits, dim = 2)
        
        if B_aug>B and args.aug_method == 'LFFA':

            soft_targets = torch.zeros_like(all_logits[B:]).squeeze(1)


            soft_targets.scatter_(1, aug_labels[B:].view(-1, 1), 1- args.aug_para3)

            soft_targets.scatter_(1, aug_labels[B:].view(-1, 1)+ args.n_classes, args.aug_para3)
            log_probs = F.log_softmax(all_logits[B:].squeeze(1), dim=1)
            loss2 = -(soft_targets * log_probs).sum(dim=1).mean()

            loss = (((1- args.aug_para2)*loss_fn(logits.squeeze(1), labels)) + args.aug_para2*loss2)
        elif B_aug>B:
            if args.aug_method in ['mhead']:
                logits_hk = all_logits.reshape(B, int(B_aug/B), 1, logits.shape[-1])  # [H, K, 1, C]
                logits_ens = logits_hk.squeeze(2).mean(dim=1) 
                loss = loss_fn(logits_ens, labels)
            else:
                loss = loss_fn(all_logits.squeeze(1), aug_labels)
        else:
            loss = loss_fn(logits.squeeze(1), labels)
        loss_value = loss.item()
        
        train_loss += loss_value
        #if (batch_idx + 1) % 20 == 0:
        #    print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
        allY_hat = torch.argmax(all_logits, dim = 2)
        if aug_labels is not None:
            acc_logger.log(allY_hat,  aug_labels)
            error = calculate_error(allY_hat, aug_labels)
            train_num = allY_hat.shape[0]
        else:
            acc_logger.log(Y_hat, labels)
            error = calculate_error(Y_hat, labels)
            train_num = Y_hat.shape[0]
        train_error += error
        train_sum += train_num

        loss.backward()
        # step
        optimizer.step()
        

    train_loss /= len(aug_train_loader)
    train_error /= train_sum

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            if acc is not None:
                writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def validate_linear(cur, epoch, model, val_loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None,model_base = None,
             num_region =None,model_base1 = None,args = None, return_auc = False):
    model.eval()
    #args.infer_time = 3
    if model_base is not None:
        model_base.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(val_loader.dataset), n_classes))
    labels = np.zeros(len(val_loader.dataset))
    patient_results = {}
    with torch.no_grad():
        #print(len(val_features),len( val_labels),len(val_slide_ids))
        for batch_idx, (data, label, slide_id) in enumerate(val_loader):
            
            data = data.to(device)
            
            B = data.shape[0]
            target = data[:,0,:]
            data = data[:,1:,:]
            data = data.to(device)
            target = target.to(device)
            label = label.to(device)
            logits, aug_labels= model(data,target,label,infer_time = args.infer_time)
            logits = logits[:,:,:args.n_classes]
            B_aug = logits.shape[0]
            Y_hat = torch.argmax(logits, dim = 2)

            acc_logger.log(Y_hat, label)
            Y_prob = F.softmax(logits.squeeze(1), dim = 1).squeeze(0)
            if (args.infer_time > 0 and args.aug_method in ['LFFA','FSCIL','FATL']) or args.aug_method in ['mhead']:
                Y_prob = Y_prob.mean(dim = 0)
                if args.aug_method in ['mhead']:
                    loss = loss_fn(logits.mean(dim = 0), label)
                else:
                    loss = loss_fn(logits.mean(dim = 0), label)

            else:
                loss = loss_fn(logits.squeeze(1), label)
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            if args.infer_time > 0 and args.aug_method in ['LFFA','FSCIL','FATL']:
                error = calculate_error(Y_hat, aug_labels) / (args.infer_time+1)
            elif args.aug_method in ['mhead']:
                error = calculate_error(Y_hat, aug_labels) / (B_aug / B)
            else:
                error = calculate_error(Y_hat, label)
            val_error += error
            #print(batch_idx)
    val_error /= len(val_loader.dataset)
    val_loss /= len(val_loader.dataset)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
    if return_auc:
        print('\n Finally Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    else:
        print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     
    print()
    #print('model.alpha:', torch.sigmoid(model.alpha)[:10])

    if return_auc:
        if n_classes == 2:
            auroc = auc
            if prob.ndim > 1 and prob.shape[1] > 1: 
                y_pred = np.argmax(prob, axis=1)
            else: 
                y_pred = (prob >= 0.5).astype(int)
            y_pred_cls = np.argmax(y_pred, axis=1)
            balanced_acc = balanced_accuracy_score(labels, y_pred_cls)
            return patient_results, 1-balanced_acc, auroc, acc_logger
        else:

            auroc = auc
            y_pred = np.argmax(prob, axis=1)     # (566,)
            weighted_f1 = f1_score(labels, y_pred, average='weighted')
            #weighted_f1 = f1_score(labels, prob, average='weighted')
            if prob.ndim > 1 and prob.shape[1] > 1:   
                y_pred = np.argmax(prob, axis=1)
            else: 
                y_pred = (prob >= 0.5).astype(int)
            y_pred_cls = np.argmax(y_pred, axis=1)
            balanced_acc = balanced_accuracy_score(labels, y_pred_cls)
            return patient_results, 1 - balanced_acc, weighted_f1, acc_logger

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,model_base = None,num_region = None,args = None):   
    model.train()
    if model_base is not None:
        model_base.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    #print(n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label,corrds,N_values) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        if model_base is not None:
            N_values = N_values.to(device)#data N*1024  

            for i in range(corrds.shape[0]):
                diffs = np.linalg.norm(corrds[i][1:] - corrds[i][:-1], axis=1)
                count_512 = np.sum(diffs == 512)
                count_1024 = np.sum(diffs == 1024)
                patch_size = 512 if count_512 > count_1024 else 1024
                corrds[i] = corrds[i] // patch_size
            corrds = corrds.to(device)        

            corrds = corrds#.half()
            
            data = data#.half()
            #N_values = N_values.half()
            #with torch.amp.autocast('cuda'):
            if args.model_name in ['']:
                regions, regions_corrds, regions_num,min_coords_vals = get_relation_V1(corrds,args.max_window_size[0],N_values,threshold=5)
                max_roi_num = regions_corrds.shape[1]
                regions, regions_corrds, regions_num, corrds,N_values,min_coords_vals = regions.to(device), regions_corrds.to(device), \
                                    regions_num.to(device),corrds.to(device),N_values.to(device),min_coords_vals.to(device)
                _, data,num_adapt_region, results_dict, rna_emb, caption_emb = model_base(data,max_roi_num, regions, regions_corrds, 
                                                                                    regions_num, N_values,corrds, min_coords_vals = min_coords_vals, 
                                                                                    reture_wsi = True)
                if data.shape[0] == 1:
                    data = data[:,:num_adapt_region[0],:]
                else:
                    raise NotImplementedError
            elif args.model_name in ['']:
                    corrds,N_values = corrds.to(device),N_values.to(device)
                    min_vals, _ = torch.min(corrds, dim=1)
                    max_vals, _ = torch.max(corrds, dim=1)
                    max_x, max_y = ((max_vals - min_vals) // args.max_window_size[0] + 1).max(dim=0)[0]
                    max_roi_num =  int(max_x * max_y)
                    _, data,num_adapt_region, results_dict, rna_emb, caption_emb = model_base(data,  
                                                                                            N_values, corrds,max_roi_num = max_roi_num, 
                                                                                            reture_wsi = True)  
                    if data.shape[0] == 1:
                        data = data[:,:num_adapt_region[0],:]
                    else:
                        raise NotImplementedError

        data = data.squeeze(0)

        logits, _, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value

        error = calculate_error(Y_hat, label)
        train_error += error

        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None,model_base = None,
             num_region =None,model_base1 = None,args = None):
    model.eval()
    if model_base is not None:
        model_base.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label,corrds,N_values) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if model_base is not None:
                data, label, N_values = data.to(device), label.to(device), N_values.to(device)#data N*1024  

                for i in range(corrds.shape[0]):
                    diffs = np.linalg.norm(corrds[i][1:] - corrds[i][:-1], axis=1)
                    count_512 = np.sum(diffs == 512)
                    count_1024 = np.sum(diffs == 1024)
                    patch_size = 512 if count_512 > count_1024 else 1024
                    corrds[i] = corrds[i] // patch_size
                corrds = corrds.to(device)

                corrds = corrds#.half()
                max_roi_num = int((data.shape[1]**(5/7)/num_region)+1)
                data = data#.half()
                #N_values = N_values.half()
                #with torch.amp.autocast('cuda'):

            data = data.squeeze(0)

            logits, Y_prob, Y_hat, _, _ = model(data)   

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
    
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    
    return False

def summary(model, loader, n_classes,model_type,patch_size = 112,max_window_size = 10,model_base = None,num_region = None,args = None):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    if model_base is not None:
        model_base.eval()
    model.eval()

    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader.dataset), n_classes))
    all_labels = np.zeros(len(loader.dataset))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data = data.to(device)
        B = data.shape[0]
        target = data[:,0,:]
        data = data[:,1:,:]
        data = data.to(device)
        target = target.to(device)
        label = label.to(device)
        with torch.inference_mode():
            logits, aug_labels= model(data,target,label,infer_time = args.infer_time)
            logits = logits[:,:,:args.n_classes]
            B_aug = logits.shape[0]
        Y_hat = torch.argmax(logits, dim = 2)
        acc_logger.log(Y_hat, label)

        Y_prob = F.softmax(logits.squeeze(1), dim = 1).squeeze(0)
        if (args.infer_time > 0 and args.aug_method in ['LFFA','FSCIL','FATL']) or args.aug_method in ['mhead']:
            Y_prob = F.softmax(logits.mean(dim = 0), dim = 1)
            #Y_prob = Y_prob.mean(dim = 0)
        all_probs[batch_idx] = Y_prob.cpu().numpy()
        all_labels[batch_idx] = label.item()
        
        if args.infer_time > 0 and args.aug_method in ['LFFA','FSCIL','FATL']:
            error = calculate_error(Y_hat, aug_labels) / (args.infer_time+1)
        elif args.aug_method in ['mhead']:
            error = calculate_error(Y_hat, aug_labels) / (B_aug /B)
        else:
            error = calculate_error(Y_hat, label)
        test_error += error
        #patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'label': label.item()}})

    test_error /= len(loader.dataset)

    y_pred = np.argmax(all_probs, axis=1) 
    balanced_acc = balanced_accuracy_score(all_labels, y_pred)
    test_error = 1 - balanced_acc
    if all_probs.shape[1] == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:

        auc = f1_score(all_labels, y_pred, average='weighted')

    return None, test_error, auc, acc_logger
