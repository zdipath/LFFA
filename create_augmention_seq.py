
import pdb
import os
import re
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits,Generic_WSI_Dataset
import argparse
import numpy as np
import torch
parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')

parser.add_argument('--experiment_target', type = str,default='PTEN',
					choices=['DIED','RAS','BRAF','KRAS','3cls_subtype','7cls_subtype','','KRAS','STK11','KEAP1','PTEN','CTNNB1','SMAD4','CASP8',
              				'VHL','SETD1B','ARID1A','APC','ACVR2A','OT-46-FFPE','OT-46','EGFR','TP53','MSI','BRAF','KRAS_1','ER','PR','HER2',
			  				'PIK3CA','IDH','coarse_subtype','fine_subtype','tumor','subtyping','grading',
							'BAP1','PBRM1','SETD2' ,
                            '1-BRCA','2-BRCA','3-BRCA','4-BRCA','5-BRCA','6-BRCA'],
					help='name about experiment')
parser.add_argument('--task', type = str,default='t1_subtype',choices=['died','t1_subtype','t1_tumor','t1_gene',
                                                                       't2_segmen','t2_combine','t2_cross',
                                                                       't3_zero_shot', 't1_os'],
					help='name about experiment')
parser.add_argument('--dataset', type = str,default='cptac',#'EBRAIN' 'cptac' 'MUT'
					help='name about dataset')
parser.add_argument('--subdataset', type = str, #default='GBM',
					help='name about dataset')
parser.add_argument('--data_root_dir', type=str, default='/data_993/public_files/public_datasets/clam_extract_fea/', 
                    help='data directory')
parser.add_argument('--val_frac', type=float, default= 0.2,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.2,
                    help='fraction of labels for test (default: 0.1)')
#new about this work
parser.add_argument('--aug_task', type = str,default='imbal_aug',choices=['imbal_aug','bal_aug',
                                                                          '1-shot','2-shot','4-shot','8-shot', '16-shot','32-shot',
                                                                          '1-shot_aug_2','1-shot_aug_4','1-shot_aug_8','1-shot_aug_16','1-shot_aug_32',
                                                                                         '2-shot_aug_4','2-shot_aug_8','2-shot_aug_16','2-shot_aug_32',
                                                                                                        '4-shot_aug_8','4-shot_aug_16','4-shot_aug_32',
                                                                                                                       '8-shot_aug_16','8-shot_aug_32',
                                                                                                                                      '16-shot_aug_32',
                                                                           'no_aug'],
					help='name about augmentation_task')
parser.add_argument('--aug_method', type = str,default='wsi',choices=['wsi','patch','pixel','weight_sampler'],
					help='name about augmentation method')
#default
parser.add_argument('--split_aug_path', type = str,default='./splits_aug',
					help='name about experiment')
parser.add_argument('--split_path', type = str,default='./splits',
					help='name about experiment')
parser.add_argument('--csv_path', type = str,default='./dataset_csv',
					help='name about experiment')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--gpu', type=int, default=5)
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
args = parser.parse_args()
if args.subdataset is None:
    args.csv_path = os.path.join(args.csv_path,'%s_clean_%s_%s.csv'%(args.task,args.dataset,args.experiment_target))
else:
    args.csv_path = os.path.join(args.csv_path,'%s_clean_%s_%s_%s.csv'%(args.task,args.dataset,args.subdataset,args.experiment_target))

#['t1_subtype','t1_gene','t2_segmen','t3_zero_shot']

if args.dataset == 'pandas':
    data_formats = '.tiff'
elif args.dataset in ['DHMC_LUNG']:
    data_formats = '.tif'
elif args.dataset in ['DHMC_RCC']:
    data_formats = '.png'
elif args.dataset in ['BCNB']:
    data_formats = '.jpg'
elif args.dataset in ['EBRAIN']:
    data_formats = '.ndpi'
else:
    data_formats = '.svs'

if args.dataset in ['pandas']:
    args.suffix = '0_512'
elif args.dataset in ['DHMC-RCC','BCNB']:
    args.suffix = '0_256'
else:
    args.suffix = '0_1024'
dataset_factory = Generic_WSI_Dataset
#else:
    #dataset_factory = Generic_MIL_Dataset
if args.task in ['t1_gene','died']:
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat=True,
                            ignore=[])


elif args.task in ['t2_combine','t2_cross','t1_subtype']:
    if args.dataset == 'EBRAIN' and args.experiment_target == 'fine_subtype':
        args.n_classes=30
    elif args.dataset == 'EBRAIN' and args.experiment_target == 'coarse_subtype':
        args.n_classes=12
    elif args.dataset == 'IMP':
        args.n_classes = 3
    elif args.dataset == 'BRACS' and args.experiment_target == 'coarse_subtype':
        args.n_classes=3
    elif args.dataset == 'BRACS' and args.experiment_target == 'fine_subtype':
        args.n_classes=7
    elif args.dataset == 'TCGA' and args.experiment_target in  ['OT-46-FFPE','OT-46']:
        args.n_classes = 46
    elif args.dataset == 'NXELC' and args.experiment_target in ['subtyping']:
        args.n_classes = 3
    elif args.dataset == 'cptac' and args.experiment_target in ['OT-46']:
        args.n_classes = 10
    elif args.dataset == 'MUT' and args.task in ['t2_combine','t2_cross']:    
        args.n_classes = 2
    elif args.dataset == 'TCGA' and args.subdataset in ['RCC']:
        args.n_classes = 3
    elif args.dataset == 'TCGA' and args.subdataset in ['LUNG','BRCA']:
        args.n_classes = 2
    elif args.dataset == 'NXELC' and args.experiment_target == 'subtyping':
        args.n_classes = 3
    label_dict = {i: i for i in range(args.n_classes)}
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_dict,
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task in ['t1_tumor']:
    args.n_classes=2
    label_dict = {i: i for i in range(args.n_classes)}
    #print(args.csv_path)
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_dict,
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
elif args.task in ['t1_os']:
    if args.dataset == 'cptac_ccrcc':
        args.n_classes=8
    elif args.dataset == 'cptac_hnsc':
        args.n_classes=8
    elif args.dataset == 'cptac_luad':
        args.n_classes=8
    elif args.dataset == 'cptac_pda':
        args.n_classes=8
    elif args.dataset == 'surgen_coad':
        args.n_classes=8
    label_dict = {i: i for i in range(args.n_classes)}
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_dict,
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

        
else:
    raise NotImplementedError
if args.dataset == 'EBRAIN' and args.experiment_target in ['fine_subtype','coarse_subtype']:
    args.val_frac = 0.25
    args.test_frac = 0.25
elif args.dataset == 'TCGA' and args.experiment_target in ['OT-46-FFPE','OT-46']:
    args.val_frac = 0.12#0.15
    args.test_frac = 0.12#0.15
elif args.dataset == 'cptac' and args.experiment_target not in ['OT-46-FFPE','OT-46','tumor','subtyping']:
    args.test_frac = 0.2
    args.val_frac = 0
    args.k = 50 




if args.split_dir is None:
    if args.subdataset is None:
        args.split_dir  = os.path.join('./splits', args.task,'%s_%s_%s'%(args.dataset,args.experiment_target,args.label_frac*100))
    else:
        args.split_dir  = os.path.join('./splits', args.task,'%s_%s_%s_%s'%(args.dataset,args.subdataset,args.experiment_target,args.label_frac*100))
else:
    args.split_dir = os.path.join('./splits', args.split_dir)
    
def build_expand_df(slide_ids, add_num, split_aug_path_shot, shuffle_remainder=False, seed=42):

    rows = []
    rng = np.random.RandomState(seed)

    add_num = add_num.tolist() if isinstance(add_num, torch.Tensor) else list(add_num)

    for cls_idx, (sids, m) in enumerate(zip(slide_ids, add_num)):
        sids = list(sids)
        n = len(sids)
        m = int(m)

        if n == 0:
            if m != 0:
                raise ValueError(f"Class {cls_idx} has no samples but requires expansion of {m}.")
            continue


        if m == 0:
            for sid in sids:
                rows.append((sid, 0))
            continue

        if m < 0:
            remove = -m
            if remove > n:
                raise ValueError(f"Class {cls_idx}: cannot remove {remove} samples from {n} available.")


            counts = np.zeros(n, dtype=int)

            order = np.arange(n)
            if shuffle_remainder:
                rng.shuffle(order)

            drop_idx = order[:remove]
            counts[drop_idx] = -1

            assert int(counts.sum()) == m, f"class {cls_idx}: sum {counts.sum()} != {m}"

            rows.extend(zip(sids, counts.tolist()))
            continue

        base, rem = divmod(m, n)
        counts = np.full(n, base, dtype=int)

        if rem > 0:
            order = np.arange(n)
            if shuffle_remainder:
                rng.shuffle(order)
            counts[order[:rem]] += 1


        assert int(counts.sum()) == m, f"class {cls_idx}: sum {counts.sum()} != {m}"

        rows.extend(zip(sids, counts.tolist()))

    df = pd.DataFrame(rows, columns=["slide_id", "expand_count"])
    return df
def build_expand_df_nested(
    slide_ids,
    add_num,
    split_aug_path_shot,
    shuffle_remainder=False,
    seed=42,
):

    rows = []
    rng = np.random.RandomState(seed)

    add_num = add_num.tolist() if isinstance(add_num, torch.Tensor) else list(add_num)
    prev_keep_set = None
    if split_aug_path_shot is not None:
        prev_df = pd.read_csv(split_aug_path_shot)
        if "slide_id" not in prev_df.columns or "expand_count" not in prev_df.columns:
            raise ValueError(
                f"Invalid prev df at {split_aug_path_shot}: must contain columns slide_id, expand_count"
            )
        prev_keep_set = set(prev_df.loc[prev_df["expand_count"] >= 0, "slide_id"].astype(str).tolist())

    for cls_idx, (sids, m) in enumerate(zip(slide_ids, add_num)):
        sids = list(map(str, sids))
        n = len(sids)
        m = int(m)

        if n == 0:
            if m != 0:
                raise ValueError(f"Class {cls_idx} has no samples but requires expansion of {m}.")
            continue


        locked = set()
        if prev_keep_set is not None:
            locked = set([sid for sid in sids if sid in prev_keep_set])


        if m == 0:
            for sid in sids:
                rows.append((sid, 0))
            continue


        if m < 0:
            remove = -m
            if remove > n:
                raise ValueError(f"Class {cls_idx}: cannot remove {remove} samples from {n} available.")


            free_candidates = [i for i, sid in enumerate(sids) if sid not in locked]
            if prev_keep_set is not None and remove > len(free_candidates):
                raise ValueError(
                    f"Class {cls_idx}: nested-shot requires keeping {len(locked)} previous samples, "
                    f"but need to remove {remove} from only {len(free_candidates)} removable samples."
                )

            counts = np.zeros(n, dtype=int)  

            cand_idx = np.array(free_candidates, dtype=int)
            if shuffle_remainder:
                rng.shuffle(cand_idx)

            drop_idx = cand_idx[:remove]
            counts[drop_idx] = -1

            assert int(counts.sum()) == m, f"class {cls_idx}: sum {counts.sum()} != {m}"
            rows.extend(zip(sids, counts.tolist()))
            continue


        base, rem = divmod(m, n)
        counts = np.full(n, base, dtype=int)

        if rem > 0:
            order = np.arange(n)
            if shuffle_remainder:
                rng.shuffle(order)
            counts[order[:rem]] += 1

        assert int(counts.sum()) == m, f"class {cls_idx}: sum {counts.sum()} != {m}"
        rows.extend(zip(sids, counts.tolist()))

    df = pd.DataFrame(rows, columns=["slide_id", "expand_count"])
    return df

if __name__ == '__main__':
    #python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --k 10
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    print(args)
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    if args.subdataset is None:
        split_aug_path  = os.path.join(args.split_aug_path, args.task,'%s_%s'%(args.dataset,args.experiment_target),args.aug_task)
    else:
        split_aug_path  = os.path.join(args.split_aug_path, args.task,'%s_%s_%s'%(args.dataset,args.subdataset,args.experiment_target),args.aug_task)
    os.makedirs(split_aug_path, exist_ok=True)
    split_aug_path_shot = None
    for i in folds:
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        num_class = args.n_classes
        slide_ids = train_dataset.slide_cls_ids_name
        if args.aug_task in ['imbal_aug'] and args.k>10:
            num_per_class = torch.tensor([len(x) for x in slide_ids], dtype=torch.long)
            add_num = num_per_class.max() - num_per_class
            #add_num = torch.min(num_per_class, add_num)
            #add_num = add_num.clamp(max=30)
        elif args.aug_task in ['imbal_aug'] and args.k<11:
            num_per_class = torch.tensor([len(x) for x in slide_ids], dtype=torch.long)
            num_classes = len(num_per_class)
            total_num = num_per_class.sum()

            add_num = num_per_class.max() - num_per_class
        elif 'shot' in args.aug_task:

            num_per_class = torch.tensor([len(x) for x in slide_ids], dtype=torch.long)

            m_aug = re.match(r'(\d+)-shot_aug_(\d+)', args.aug_task)
            m_plain = re.match(r'(\d+)-shot$', args.aug_task)

            if m_aug is not None:

                base_shot = int(m_aug.group(1))  # 2
                target = int(m_aug.group(2))     # 4

                if target % base_shot != 0:
                    raise ValueError(f"Invalid aug_task {args.aug_task}: target ({target}) "
                                    f"must be a multiple of base_shot ({base_shot}).")


                dup_factor = target // base_shot      # 2
                aug_times_per_sample = dup_factor - 1 


                target_per_class = torch.full_like(num_per_class, base_shot) #N_class* num_shot


                add_num = target_per_class - num_per_class

                if base_shot != 1:
                    split_aug_path_shot = '/'.join(split_aug_path.split('/')[:-1])
                    brf_shot = str(int(base_shot/2))
                    split_aug_path_shot = os.path.join(split_aug_path_shot,'%s-shot'%(brf_shot),'augmentaion_num_fold_{}.csv'.format(i))
            elif m_plain is not None:
                shot = int(m_plain.group(1))
                target_per_class = torch.full_like(num_per_class, shot)
                add_num = target_per_class - num_per_class
                if shot != 1:
                    split_aug_path_shot = '/'.join(split_aug_path.split('/')[:-1])
                    brf_shot = str(int(shot/2))
                    split_aug_path_shot = os.path.join(split_aug_path_shot,'%s-shot'%(brf_shot),'augmentaion_num_fold_{}.csv'.format(i))
            else:
                raise ValueError(f"Unrecognized aug_task format: {args.aug_task}")

        print('Fold %d:'%(i))
        for j in range(num_class):
            print('The number of adding data in class %d: %d' % (j, add_num[j]))
        print('\n')
        if 'shot' in args.aug_task:
            df = build_expand_df_nested(slide_ids, add_num,split_aug_path_shot, shuffle_remainder=True, seed=args.seed)
        else:
            df = build_expand_df(slide_ids, add_num,split_aug_path_shot, shuffle_remainder=True, seed=args.seed)
        

        '''
        ['imbal_aug','bal_aug','2-shot','4-shot','8-shot', '16-shot',
            '2-shot_aug_4','2-shot_aug_8','2-shot_aug_16',
            '4-shot_aug_8','4-shot_aug_16',
            '8-shot_aug_16','no_aug'],'''
        
        df.to_csv(os.path.join(split_aug_path, 'augmentaion_num_fold_{}.csv'.format(i)), index=False)



