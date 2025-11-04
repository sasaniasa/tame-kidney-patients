import sys
sys.path.append('../src')

import warnings

# Global suppression
warnings.filterwarnings("ignore", message="Parsing dates in .* when dayfirst=True was specified")
warnings.filterwarnings("ignore", category=FutureWarning)


from preprocessing import *
import tame
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import myloss
import function
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda(obj):
    if isinstance(obj, torch.nn.Module):
        return obj.cuda()
    elif isinstance(obj, torch.Tensor):
        return obj.cuda()
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).cuda()
    
def index_value(data):
    index = data // (CONFIG['split_num'] + 1)
    value = data % (CONFIG['split_num'] + 1)
    index = Variable(_cuda(index)) #Variable(_cuda(index))
    value = Variable(_cuda(value)) #Variable(_cuda(value))
    return [index, value]

def move_batch_to_device(batch, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

def compute_corr_per_feature(pred, label, mask):
    B, T, F = pred.shape
    corr_scores = []

    for f in range(F):
        y_true = label[:, :, f][mask[:, :, f] == 1]
        y_pred = pred[:, :, f][mask[:, :, f] == 1]

        if len(y_true) < 2:
            corr_scores.append(np.nan)
            continue

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        corr_scores.append(corr)

    return np.array(corr_scores)


def train_eval(data_loader, net, loss=None, epoch= None, optimizer= None, best_metric= None, phase='train'):
    print(phase)
    lr = 0.001
    if phase == 'train':
        net.train()
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr
    else:
        net.eval()

    loss_list, pred_list, label_list, mask_list, pad_mask_list = [], [], [], [], []
    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        #print(batch.keys())

        data = batch['input_ts_features']
        label_data = batch['output_ts_features']
        value_mask = batch['value_mask']
        static_num_data = batch['static_numerical_features']
        static_cat_data = batch['static_categorical_features']
        pre_input, pre_time, post_input, post_time = batch['pre_input'], batch['pre_time'], batch['post_input'], batch['post_time'] 
        pad_mask = batch['mask']

        #print("mask before", value_mask.shape)

        data = index_value(data)
        label_data = Variable(_cuda(label_data))#Variable(label_data) 
        #print(label_data.shape)
        value_mask = Variable(_cuda(value_mask)) # Variable(value_mask)
        static_num_data = Variable(_cuda(static_num_data)) #Variable(static_num_data)
        static_cat_data = Variable(_cuda(static_cat_data)) # Variable(static_cat_data)
        pre_input = index_value(pre_input)
        post_input = index_value(post_input)
        pre_time = Variable(_cuda(pre_time)) #Variable(pre_time)
        post_time = Variable(_cuda(post_time)) #Variable(post_time)
        neib = [pre_input, pre_time, post_input, post_time]
        pad_mask = Variable(_cuda(pad_mask)) # Variable(pad_mask)
        #print(mask)

        B, T, D1 = label_data.shape
        D2 = static_num_data.shape[1]
        D3 = static_cat_data.shape[1]
        # Expand static data to shape [B, T, D]
        static_num_exp = static_num_data.unsqueeze(1).expand(B, T, D2)         # [B, T, D2]
        static_cat_exp = static_cat_data.unsqueeze(1).expand(B, T, D3) # [B, T, D3]

        real_data = torch.cat([label_data, static_num_exp, static_cat_exp], dim=2)

        real_data = Variable(_cuda(real_data)) #Variable(_cuda(real_data))

        output = net(data, neib, static_num_data, static_cat_data, mask=pad_mask) # ([1, 1745, 15])

        #print('output', output.shape)
        #print('label', label_data.shape)
        #print('value_mask', value_mask.shape)
        print("Output calculation finished")

        loss_output = loss(output, label_data, value_mask, pad_mask) # for training
        pred_list.append(output.data.cpu().numpy())
        loss_list.append(loss_output.data.cpu().numpy())
        label_list.append(label_data.data.cpu().numpy())
        mask_list.append(value_mask.data.cpu().numpy()) # value mask 
        pad_mask_list.append(pad_mask.data.cpu().numpy())

        if phase == 'train':
            optimizer.zero_grad()
            loss_output.backward()
            # Clip gradients to avoid explosion
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()


    pred = np.concatenate(pred_list, 0)
    label = np.concatenate(label_list, 0)
    mask = np.concatenate(mask_list, 0)
    pad_mask = np.concatenate(pad_mask_list, 0)


    r2s = compute_corr_per_feature(pred, label, mask)
    for i, r2 in enumerate(r2s):
        print(f"Feature {i}: R² = {r2:.4f}")
    #print('pred', pred.shape)
    #print('label', label.shape)
    #print('mask', mask.shape)
    print("Calculating the nRMSE Loss ...")
    metric_list = function.compute_nRMSE(pred, label, mask, pad_mask) # for evaluation
    avg_loss = np.mean(loss_list)

    print('loss: {:3.4f} \t'.format(avg_loss))
    print('metric: {:s}'.format('\t'.join(['{:3.4f}'.format(m) for m in metric_list[:2]])))

    model_dir = 'model_checkpoints'
    os.makedirs(model_dir, exist_ok=True)
    metric = metric_list[0]
    if phase == 'valid' and (best_metric[0] == 0 or best_metric[0] > metric):
        best_metric = [metric, epoch]
        best_r2s = r2s
        function.save_model({'model': net, 'epoch':epoch, 'best_metric': best_metric, 'best_r2s':best_r2s}, model_dir)

    torch.cuda.empty_cache()

    return output, label_data, avg_loss, best_metric, pred, label, mask, pad_mask

# === Compute and display feature-wise RMSE after test_eval ===

def compute_feature_nrmse(pred, label, mask):
    """
    Compute feature-wise nRMSE (normalized Root Mean Squared Error).
    pred, label, mask should all be numpy arrays of shape (B, T, F).
    Returns an array of shape (F,) with nRMSE per feature.
    """
    B, T, F = pred.shape
    nrmse_list = []

    for f in range(F):
        y_true = label[:, :, f][mask[:, :, f] == 1]
        y_pred = pred[:, :, f][mask[:, :, f] == 1]

        if len(y_true) < 2:
            nrmse_list.append(np.nan)
            continue

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mean_abs = np.mean(np.abs(y_true))  # normalization factor

        if mean_abs == 0:
            nrmse_list.append(np.nan)
        else:
            nrmse_list.append(rmse / mean_abs)

    return np.array(nrmse_list)

def compute_corr_per_feature(pred, label, mask):
    B, T, F = pred.shape
    corr_scores = []

    for f in range(F):
        y_true = label[:, :, f][mask[:, :, f] == 1]
        y_pred = pred[:, :, f][mask[:, :, f] == 1]

        if len(y_true) < 2:
            corr_scores.append(np.nan)
            continue

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        corr_scores.append(corr)

    return np.array(corr_scores)

def shuffle_time_within_patient(ts_df):
    'for abaltion study'
    return ts_df.groupby('patient_id').apply(
        lambda df: df.sample(frac=1, random_state=42).reset_index(drop=True)
    ).reset_index(drop=True)



def main():
    dfs = get_dfs()
    biopsy_df = dfs['biopsy']
    static_df= create_static_df(dfs)
    vital_df = create_vitals_df(dfs, static_df)
    med_df = create_medication_df(dfs, static_df)
    lab_df = create_lab_values_df(dfs, static_df)

    ts_data = create_ts_df(vital_df, lab_df, med_df, merge_lab=True, merge_med=True)
    print(ts_data.columns)

    # Create dataset
    full_dataset = NephroDataset(static_df, ts_data, biopsy_df)
    categorical_cardinalities = full_dataset.categorical_cardinalities

    datapoints_limit = len(full_dataset)
    print(f"Datapoints limit: {datapoints_limit}")

    dataset = Subset(full_dataset, indices=list(range(datapoints_limit)))

    # Define split proportions
    train_ratio = 0.6
    val_ratio = 0.25
    
    total = len(dataset)

    # Compute sizes
    train_size = max(1, int(train_ratio * total))
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    # Adjust in case of very small dataset
    if test_size == 0 and val_size > 0:
        test_size = 1
        val_size = max(0, val_size - 1)
    elif test_size == 0 and val_size == 0:
        test_size = 0
        val_size = 0
        train_size = total

    print(f"Splitting {total} samples → Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = CONFIG['batch_size']

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    print("Train and validation dataloaders created")

    torch.cuda.empty_cache()
    torch.set_num_threads(1)
    _ = torch.tensor([0.0]).cuda()
    print("✅ CUDA context warmed up")

    net = _cuda(tame.AutoEncoder(categorical_cardinalities=categorical_cardinalities))
    print("Model is initialized!")
    #net = (net)
    #print("Model on GPU!")
    loss_fn = _cuda(myloss.MSELoss(loss='both'))
    print("Loss function is initialized")

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    print("Model and loss function are initialized")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Track losses
    train_losses = []
    val_losses = []

    best_metric = [0, 0]  # Initialize best metric and epoch

    for epoch in range(1, 35 + 1):
        print(f"\n===== Epoch {epoch} =====")

        # --- Training Phase ---
        _, _, train_loss, _, _, _, _ ,_= train_eval(
            train_dataloader,
            net,
            loss=loss_fn,
            epoch=epoch,
            optimizer=optimizer,
            best_metric=best_metric,
            phase='train'
        )
        train_losses.append(train_loss)

        # --- Validation Phase ---
        with torch.no_grad():
            _, _, val_loss, best_metric, _, _, _, _ = train_eval(
                val_dataloader,
                net,
                loss=loss_fn,
                epoch=epoch,
                optimizer=None,
                best_metric=best_metric,
                phase='valid'
            )
            scheduler.step(val_loss)
            val_losses.append(val_loss)
            
    print(f"Best metric: {best_metric}")


    # Plotting the losses
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 35 + 1), train_losses, label='Training Loss')
    plt.plot(range(1, 35 + 1), val_losses, label='Validation Loss')
    plt.xticks(ticks=range(0, 36, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # save plot
    plt.savefig('loss_plot.png')
    plt.show()

    print("\n===== Final Evaluation on Training and Test Set =====")

    # Training predictions
    with torch.no_grad():
        _, _, _, _, train_pred, train_label, train_mask, train_pad_mask = train_eval(
            train_dataloader,
            net,
            loss=loss_fn,
            epoch=epoch,
            optimizer=None,
            best_metric=None,
            phase='test'
        )

    # Test predictions
    with torch.no_grad():
        _, _, test_loss, _, test_pred, test_label, test_mask, test_pad_mask = train_eval(
            test_dataloader,
            net,
            loss=loss_fn,
            epoch=epoch,
            optimizer=None,
            best_metric=None,
            phase='test'
        )

    print(f"Test Loss: {test_loss}")

    # MSEs
    training_mse = np.mean((train_pred - train_label)**2 * train_mask) / np.mean(train_mask)
    test_mse = np.mean((test_pred - test_label)**2 * test_mask) / np.mean(test_mask)

    # Bias^2
    bias_squared = np.mean((test_pred - test_label)**2 * test_mask) / np.mean(test_mask)

    # Variance (single-run)
    variance = 0.0

    # Irreducible error
    irreducible_error = test_mse - bias_squared - variance

    # Print
    print(f"\nTraining MSE: {training_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Interpret
    if training_mse > 0.1 and test_mse > 0.1: 
        print("Interpretation: High bias (model underfitting).")
    elif training_mse < 0.1 and test_mse > (training_mse * 3):
        print("Interpretation: High variance (model overfitting).")
    else:
        print("Interpretation: Balanced or mixed bias/variance.")

    print(f"Bias^2: {bias_squared}")
    print(f"Variance: {variance}")
    print(f"Irreducible Error: {irreducible_error:}")

    # === Compute RMSE on test predictions ===
    test_rmse = compute_feature_nrmse(test_pred, test_label, test_mask)

    # === Compute test Pearson correlation per feature ===
    test_corrs = compute_corr_per_feature(test_pred, test_label, test_mask)

    # === Print them alongside RMSE ===
    print("\n📊 Test Set — Feature-wise Pearson Correlation and RMSE:")
    for i, (rmse, corr) in enumerate(zip(test_rmse, test_corrs)):
        feat_name = CONFIG['ts_feat'][i] if i < len(CONFIG['ts_feat']) else f"Feature {i}"
        print(f"{feat_name:<15} | RMSE = {rmse:.4f} | r = {corr:.4f}")

     # Ablation study: Evaluate on shuffled data
    print("\n--- Ablation Study: Shuffled Evaluation ---")
    
    # 1. Get the indices for the test dataset
    test_indices = test_dataset.indices
    
    # 2. Get the original data from the full dataset
    test_data_original = full_dataset.ts_data.iloc[test_indices]
    
    # 3. Shuffle the time-series data within each patient
    test_data_shuffled = shuffle_time_within_patient(test_data_original)
    
    # 4. Create a new dataset and dataloader for the shuffled data
    shuffled_dataset = NephroDataset(full_dataset.static_df.iloc[test_indices], test_data_shuffled, biopsy_df.iloc[test_indices])
    shuffled_test_loader = DataLoader(shuffled_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # 5. Evaluate the model on the shuffled data
    _, _, test_loss_shuffled, _, pred_shuffled, label_shuffled, mask_shuffled, _ = train_eval(shuffled_test_loader, net, loss_fn, phase='test')

    # 6. Compare the results
    print("\n--- Comparison of Results ---")
    print("Metrics on Original Test Data:")
    # === Compute RMSE on test predictions ===
    test_rmse = compute_feature_nrmse(test_pred, test_label, test_mask)
    # === Compute test Pearson correlation per feature ===
    test_corrs = compute_corr_per_feature(test_pred, test_label, test_mask)
    print("nRMSE:", test_rmse)
    print("Correlation:", test_corrs)
    print("Test Loss:", test_loss)

    print("\nMetrics on Shuffled Test Data:")
    nrmse_shuffled = compute_feature_nrmse(pred_shuffled, label_shuffled, mask_shuffled)
    corr_shuffled = compute_corr_per_feature(pred_shuffled, label_shuffled, mask_shuffled)
    print("nRMSE:", nrmse_shuffled)
    print("Correlation:", corr_shuffled)
    print("Test Loss:", test_loss_shuffled)

if __name__ == "__main__":
    main()
