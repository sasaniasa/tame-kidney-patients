import numpy as np
import os
import torch
def compute_nRMSE(pred, label, mask, visit_mask=None): # visit mask == pad mask
    '''
    same as 3dmice
    '''
    assert pred.shape == label.shape == mask.shape

    #print("nans in pred", np.isnan(pred).sum())
    #print("nans in label", np.isnan(label).sum())
    #print("nans in mask", np.isnan(mask).sum())

    if visit_mask is not None:
        # Convert to boolean if needed
        visit_mask = visit_mask.astype(bool)

        # Expand from [B, T] → [B, T, 1] then broadcast to [B, T, F]
        visit_mask = np.expand_dims(visit_mask, axis=-1)
        visit_mask = np.broadcast_to(visit_mask, mask.shape)

        # Mark padded positions in mask as -1 (ignored in metric)
        mask = np.where(visit_mask, mask, -1)

    missing_indices = mask == 1
    missing_pred = pred[missing_indices]
    missing_label = label[missing_indices]
    missing_rmse = np.sqrt(((missing_pred - missing_label) ** 2).mean())

    init_indices = mask==0
    init_pred = pred[init_indices]
    init_label = label[init_indices]
    init_rmse = np.sqrt(((init_pred - init_label) ** 2).mean())

    metric_list = [missing_rmse, init_rmse]
    #print(pred.shape[2]) #15
    for i in range(pred.shape[2]):
        apred = pred[:,:,i]
        alabel = label[:,:, i]
        amask = mask[:,:, i]

        mrmse, irmse = [], []
        for ip in range(len(apred)):
            ipred = apred[ip]
            ilabel = alabel[ip]
            imask = amask[ip]

            x = ilabel[imask>=0]
            if len(x) == 0:
                continue

            minv = ilabel[imask>=0].min()
            maxv = ilabel[imask>=0].max()

            if maxv == minv:
               #print(f"Feature {i} has constant values: min = {minv}, max = {maxv}")
               default_rmse = 0  # or np.nan
               irmse.append(default_rmse)
               mrmse.append(default_rmse)
               continue

            init_indices = imask==0
            init_pred = ipred[init_indices]
            init_label = ilabel[init_indices]

            missing_indices = imask==1
            missing_pred = ipred[missing_indices]
            missing_label = ilabel[missing_indices]

            assert len(init_label) + len(missing_label) >= 2

            if len(init_pred) > 0:
                init_rmse = np.sqrt((((init_pred - init_label) / (maxv - minv)) ** 2).mean())
                irmse.append(init_rmse)

            if len(missing_pred) > 0:
                missing_rmse = np.sqrt((((missing_pred - missing_label)/ (maxv - minv)) ** 2).mean())
                mrmse.append(missing_rmse)

        metric_list.append(np.mean(mrmse))
        metric_list.append(np.mean(irmse))

    metric_list = np.array(metric_list)


    metric_list[0] = np.mean(metric_list[2:][::2])# average value for missing values(mask == 1)
    metric_list[1] = np.mean(metric_list[3:][::2])# average value for non missing (mask == 0)

    #print("metric list", metric_list)
    #print("len metric list", len(metric_list)) # 32
    return metric_list


def save_model(p_dict, current_dir, name='best.ckpt', folder=None):
    if folder is None:
        folder = os.path.join(current_dir, 'models')
    # name = '{:s}-snm-{:d}-snr-{:d}-value-{:d}-trend-{:d}-cat-{:d}-lt-{:d}-size-{:d}-seed-{:d}-loss-{:s}-{:d}-{:s}'.format(args.task, args.split_num, args.split_nor, args.use_value, args.use_trend, args.use_cat, args.last_time, args.embed_size, args.seed, args.loss, args.time, name)
    # name = '{:s}-{:s}-{:d}-variables-{:d}{:d}{:d}-{:s}'.format(args.dataset, args.model, len(args.name_list), args.use_ta, args.use_ve, args.use_mm, name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    model = p_dict['model']
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    all_dict = {
            'epoch': p_dict['epoch'],
            'best_metric': p_dict['best_metric'],
            'state_dict': state_dict,
            'best r2s': p_dict['best_r2s'],
            }
    torch.save(all_dict, os.path.join(folder, name))

def load_model(p_dict, model_file):
    all_dict = torch.load(model_file)
    p_dict['epoch'] = all_dict['epoch']
    # p_dict['args'] = all_dict['args']
    p_dict['best_metric'] = all_dict['best_metric']
    # for k,v in all_dict['state_dict'].items():
    #     p_dict['model_dict'][k].load_state_dict(all_dict['state_dict'][k])
    p_dict['model'].load_state_dict(all_dict['state_dict'])