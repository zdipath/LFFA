import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_armil import ARMIL_atten,get_relation
import pdb
import os
import pandas as pd
from torch.utils.data import DataLoader, Subset,ConcatDataset
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from models.model_transmil import TransMIL
from models.model_abmil import ABMIL
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
def initiate_model(args, ckpt_path, device='cuda'):
    #print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type =='abmil':
        model = ABMIL(n_classes=args.n_classes,embed_dim=args.embed_dim)
    elif args.model_type == 'transmil':
        model = TransMIL(n_classes=args.n_classes,embed_dim=args.embed_dim)
    elif args.model_type == 'lwmil':
        
        args.max_window_size=[7]
        #model = ARMIL(n_classes=args.n_classes,patch_size = args.step_size,embed_dim=args.embed_dim, num_heads=8,depth = args.depth,max_window_size=args.max_window_size)
        model = ARMIL_atten(n_classes=args.n_classes,patch_size = args.step_size,embed_dim=args.embed_dim, num_heads=8,depth = args.depth, max_window_size=args.max_window_size)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    #print_network(model)

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))#,map_location='cuda'
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model
def eval_ood(test_results_list,dataset,ood_dataset_list, args, model,meth):
    print('OOD detection method:',meth)
    acc_i_sum = []
    acc_i_80_sum = []
    acc_o_sum = []
    acc_o_80_sum = []
    id_num = []
    id_80_num = []
    ood_num = []
    ood_80_num = []
    
    print_acc = ['ACC_I: ','ACC_I_80: ','ACC_O: ','ACC_O_80: ']

    
    for item_data, test_results in enumerate(test_results_list):
        loader_sum =[]
        filter_index = []
        True_ID = test_results[0]['ID'].squeeze()
        False_ID = test_results[0]['OOD'].squeeze()
        True_ID_indices = np.where(True_ID)[0]
        filter_index.append(True_ID_indices)
        filtered_dataset = Subset(dataset, True_ID_indices)
        loader = get_coords_id_loader(dataset,num_workers=0)
        id_num.append(len(loader))
        loader_sum.append(loader)
        loader_80 = get_coords_id_loader(filtered_dataset,num_workers=0)
        loader_sum.append(loader_80)
        id_80_num.append(len(loader_80))
        False_ID_indices = np.where(False_ID)[0]
        filter_index.append(False_ID_indices)
        ood_filtered_dataset = Subset(ood_dataset_list[item_data], False_ID_indices)
        #ood_loader = get_coords_id_loader(ConcatDataset([ood_dataset, dataset]))
        ood_loader = get_coords_id_loader(ood_dataset_list[item_data],num_workers=0)
        ood_num.append(len(ood_loader))
        loader_sum.append(ood_loader)
        ood_loader_80 = get_coords_id_loader(ood_filtered_dataset,num_workers=0)
        ood_80_num.append(len(ood_loader_80))
        loader_sum.append(ood_loader_80)
        t0 = time.time()
        for item_loader, loader in enumerate(loader_sum):
            if item_loader in [0,2]:
                true_label = [] #np.ones((len(loader), 1), dtype=int)*10
                pred_label = [] #np.ones((len(loader), 1), dtype=int)*10
                is_null = True
            else:
                indices = filter_index[int(item_loader//2)]
                true_label = true_label[indices]
                pred_label = pred_label[indices]
                is_null = False
            print('The value of', print_acc[item_loader],end='')
            acc, true_label, pred_label= summary_ood(model, loader, args, true_label, pred_label, is_null)
            if item_loader == 0:
                acc_i_sum.append(acc)
            elif item_loader == 1:
                acc_i_80_sum.append(acc)
            elif item_loader ==2:
                acc_o_sum.append(acc)
            elif item_loader ==3:
                acc_o_80_sum.append(acc)
        t4 = time.time()
    acc_i = sum(a * n for a, n in zip(acc_i_sum, id_num)) / sum(id_num)
    id_avg_num = int(sum(id_num))
    #acc_o
    ood_num.append(id_avg_num)
    acc_o_sum.append(acc_i)
    acc_o = sum(a * n for a, n in zip(acc_o_sum, ood_num)) / sum(ood_num)
    ood_num_str = '_'.join(map(str, ood_num))
    #acc_i_80
    acc_i_80 = sum(a * n for a, n in zip(acc_i_80_sum, id_80_num)) / sum(id_80_num)
    id_80_avg_num = int(sum(id_80_num))
    #acc_o_80
    ood_80_num.append(id_80_avg_num)
    acc_o_80_sum.append(acc_i_80)
    acc_o_80 = sum(a * n for a, n in zip(acc_o_80_sum, ood_80_num)) / sum(ood_80_num)
    ood_80_num_str = '_'.join(map(str, ood_80_num))

    return acc_i, id_avg_num, acc_o, ood_num_str, acc_i_80, id_80_avg_num, acc_o_80, ood_80_num_str

def calculate_error_numpy(Y_hat, Y):
	error = 0 if Y_hat == Y else 1
	return error
def summary_ood(model, loader, args, all_labels, all_preds, is_null):
    model.eval()
    test_error = 0.
    if is_null == False:
        test_error = np.sum(all_labels != all_preds)
        
        test_error /= len(loader)
        acc = 1-test_error
        print(acc)
        return  acc,all_labels, all_preds
    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))
    ood_label = False
    for batch_idx,batch_data in (enumerate(loader)):
        #if batch_idx ==2:
        #    break
        data, label, coords, id = batch_data#[21816, 1024]
        #print(batch_idx)
        id = id[0]
        data, label = data.cuda(), label.cuda()
        if label == -1:
            ood_label = True
            test_error += 1
        else:
            with torch.no_grad():
                _, Y_prob, Y_hat, _, _ = model(data)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            all_preds[batch_idx] = Y_hat.item()
            error = calculate_error(Y_hat, label)
            test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    '''
    if ood_label:
        auc_score = -1
    elif len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            
            
            auc_score = np.nanmean(np.array(aucs))'''

    acc = 1-test_error
    print(acc)

    return  acc, all_labels, all_preds


def confusion_matrix_computing(output_path,loader,model,classes,name):
    labels = []
    pre_labels = []
    t1 = time.time()
    output_path1 = os.path.join(output_path,name+'.csv')
    pred = torch.zeros(len(loader),classes)
    for item_data,batch_data in enumerate(loader):
        #if item_data ==5:
        #    break
        data, target, coords, id = batch_data#[21816, 1024]
        data= data.to(device)
        labels.append(target[0])
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
            pre_labels.append(Y_hat[0][0].detach().cpu())
            pred[item_data] = Y_prob.detach().cpu()
    print(time.time()-t1)
    labels = np.array(labels)
    pre_labels = np.array(pre_labels)
    cm = confusion_matrix(labels, pre_labels)
    if name == 'id':
        cm_df = pd.DataFrame(cm, index=np.unique(labels), columns=np.unique(pre_labels))
    else:
        cm_df = pd.DataFrame(cm, index=[0,1,-1], columns=[0,1,-1])

    cm_df.to_csv(output_path1, index=True)
    pred_np = pred.numpy()
    pred_df = pd.DataFrame(pred_np)
    output_path2 = os.path.join(output_path,name+'pred.csv')
    pred_df.to_csv(output_path2, index=False)
    
def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    if args.model_type in ['lwmil']:
        dataset.load_from_h5(True)
        dataset.return_slideid()
        loader = get_coords_id_loader(dataset,training = False, weighted = False, batch_size=args.batch_size,num_workers=0)
    else:

        loader = get_split_loader(dataset, testing = args.testing)
    # loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader.dataset), args.n_classes))
    all_labels = np.zeros(len(loader.dataset))
    all_preds = np.zeros(len(loader.dataset))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    num_sample = 0
    for batch_idx, datas in enumerate(loader):
        if args.model_type in ['lwmil']:
            (data, label, corrds, N_values) =datas
            data, label, N_values = data.to(device), label.to(device), N_values.to(device)#data N*1024  
            #data, label, N_values = data.cuda(), label.cuda(), N_values.cuda()#data N*1024  
            corrds//=args.step_size
            regions, regions_corrds, regions_num = get_relation(corrds,args.max_window_size[0],N_values)
            regions, regions_corrds, regions_num, corrds = regions.to(device), regions_corrds.to(device), regions_num.to(device),corrds.to(device)
            #regions, regions_corrds, regions_num, corrds = regions.cuda(), regions_corrds.cuda(), regions_num.cuda(),corrds.cuda()
            t0 = time.time()
            with torch.no_grad():
                logits, Y_prob, Y_hat, _, instance_dict = model(data,  regions, regions_corrds, regions_num, N_values, corrds)
                acc_logger.log_wsi(Y_hat, label)
        else:
            (data, label) =datas
            data, label = data.to(device), label.to(device)
            with torch.no_grad():
                logits, Y_prob, Y_hat, _, results_dict = model(data)
                acc_logger.log(Y_hat, label)
        slide_id = slide_ids.iloc[batch_idx]
        
        
        
        
        probs = Y_prob.cpu().numpy()

        all_probs[num_sample:num_sample+data.shape[0]] = probs
        all_labels[num_sample:num_sample+data.shape[0]] = label.detach().cpu().numpy()
        all_preds[num_sample:num_sample+data.shape[0]] = Y_hat.detach().cpu().numpy()
        num_sample += data.shape[0]
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.detach().cpu().numpy()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
