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
from config import CONFIG

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader, Subset, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dtw import compute_dtw, dist_func
inf = 1e10 

def compute_similarity_matrix(patient_ids, patient_features):
    assert len(patient_ids) == len(patient_features)
    
    n_patients = len(patient_ids)
    dist_matrix = np.zeros((n_patients, n_patients)) - 1

    print("Computing distance matrix...")
    for i in tqdm(range(n_patients)):
        dist_matrix[i, i] = 0
        for j in range(i + 1, n_patients):
            if dist_matrix[i, j] >= 0:
                continue # distance already computed

            s1 = patient_features[i]  # already normalized
            s2 = patient_features[j]  # already normalized

            distance_mat = dist_func(s1, s2)
            

            compute_dtw(distance_mat, None, dist_matrix, i, j)

    return patient_ids, dist_matrix

def extract_features_and_labels(model, data_loader):
    model.eval()

    all_embeddings = []
    all_ts_features = []
    all_num_features = []
    all_cat_features = []
    all_pids = []
    cat_np = []
    num_np = []

    death_labels = []
    graft_loss_labels = []

    graft_loss_days = []
    mort_days = []
    rej_days = []

    # First Pass: Extract features and collect all of them
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Move inputs to GPU if needed
            ts_data = index_value(batch['input_ts_features'])
            neib = [index_value(batch['pre_input']), Variable(_cuda(batch['pre_time'])), index_value(batch['post_input']), Variable(_cuda(batch['post_time']))]

            static_num = Variable(_cuda(batch['static_numerical_features']))
            static_num_features = static_scaler.inverse_transform(static_num.cpu().numpy())
            all_num_features.append(static_num_features)

            num_np.append(static_num.cpu().numpy())

            static_cat = Variable(_cuda(batch['static_categorical_features']))
            static_cat_np = static_cat.cpu().numpy()  # Convert to numpy for easier processing
            cat_np.append(static_cat_np)
            cat_decodec = []
            for i, (col, le) in enumerate(cat_scaler.items()):
                cat_decodec.append(le.inverse_transform(static_cat_np[:, i]))
            cat_decoed = np.array(cat_decodec).T.tolist()
            all_cat_features.extend(cat_decoed)

            patient_ids = batch['patient_id']
            real_data = Variable(_cuda(batch['real_ts_data']))

            pad_mask = Variable(_cuda(batch['mask']))

            seq_lens = batch['seq_len']

            
            output = model(ts_data, neib, static_num, static_cat, mask=pad_mask)

            batch_embeddings = []
            for i in range(output.shape[0]):
                seq_len = seq_lens[i].item()
                rep = output[i, :seq_len, :].cpu().numpy()  
                batch_embeddings.append(rep)

            all_embeddings.extend(batch_embeddings)

            # De-normalize
            output_real = output * (maxs - mins)[None, None, :] + mins[None, None, :]

            loss_days = batch['loss_rel_days']
            death_days = batch['death_rel_days']
            rej_days_batch= batch['rej_rel_days_list']

            batch_size = real_data.size(0)

            for i in range(batch_size):
                valid_len = seq_lens[i]

                # limit seq len to 200
                if valid_len >= 200:
                    valid_len = 200
                    
                # Extract the valid part (not padded) for this patient
                real_slice = real_data[i, :valid_len, :]      # Shape: (valid_len, 1 + feature_dim)
                output_slice = output_real[i, :valid_len, :]  # Shape: (valid_len, feature_dim)

                # Separate rel_days and real features
                rel_days = real_slice[:, 0].unsqueeze(-1)      # Shape: (valid_len, 1)
                real_features = real_slice[:, 1:]              # Shape: (valid_len, feature_dim)

                # Replace NaNs in real features with output prediction
                nan_mask = torch.isnan(real_features)
                real_features[nan_mask] = output_slice[nan_mask]

                all_ts_features.append(real_features.cpu().numpy())

                # Optional: Check if any NaNs remain
                if torch.isnan(real_features).any():
                    print("❌ NaN values found after fixing.")
                else:
                    print("✅ No NaN values after fixing.")

                rej_days_i = rej_days_batch[i]
                if rej_days_i is None or len(rej_days_i) == 0:
                    rej_days_i = rel_days[-1]  # Use the last rel_day if no rejection days are available
                rej_days.append(rej_days_i)

                loss_day = loss_days[i].item()
                if loss_day is None or np.isnan(loss_day):
                    loss_day = rel_days[-1].cpu().item()

                if death_days[i].item() is None or np.isnan(death_days[i].item()):
                    death_day = rel_days[-1].cpu().item()
                else:
                    death_day = death_days[i].item()

                graft_loss_days.append(loss_day)
                mort_days.append(death_day)

            graft_loss_label = batch['graft_loss_label'].cpu().numpy()
            death_label = batch['death_label'].cpu().numpy()
            graft_loss_labels.append(graft_loss_label)
            death_labels.append(death_label)
            all_pids.append(patient_ids)
    
    combined_ts_features = np.concatenate(all_ts_features, axis=0)
    all_ts_mean = np.nanmean(combined_ts_features, axis=0)
    all_ts_std = np.nanstd(combined_ts_features, axis=0)
    
    # Second Pass: Normalize all features using the calculated mean and std
    normalized_ts_features = []
    for ts in all_ts_features:
        ts_normalized = (ts - all_ts_mean) / (all_ts_std + 1e-8)  # Add a small epsilon to avoid division by zero
        normalized_ts_features.append(ts_normalized)
    
    #all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_num_features = np.concatenate(all_num_features, axis=0)
    all_cat_features = np.array(all_cat_features)
    cat_np = np.array(np.concatenate(cat_np, axis=0))
    num_np = np.array(np.concatenate(num_np, axis=0))
    static_vectors = np.concatenate((cat_np, num_np), axis=1)
    graft_loss_labels = np.concatenate(graft_loss_labels, axis=0).squeeze()
    death_labels = np.concatenate(death_labels, axis=0).squeeze()
    all_pids = np.concatenate(all_pids, axis=0)
    
    return (
        all_embeddings,
        normalized_ts_features, 
        all_num_features, 
        all_cat_features,
        static_vectors,
        all_pids, 
        death_labels, 
        graft_loss_labels, 
        graft_loss_days, 
        mort_days, 
        rej_days
    )


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
    index = Variable(_cuda(index))
    value = Variable(_cuda(value))
    return [index, value]

dfs = get_dfs()
biopsy_df = dfs['biopsy'].copy()
static_df= create_static_df(dfs)
vital_df = create_vitals_df(dfs, static_df)
med_df = create_medication_df(dfs, static_df)
lab_df = create_lab_values_df(dfs, static_df)
ts_data = create_ts_df(vital_df, lab_df, med_df, merge_lab=True, merge_med=True)
print(ts_data.columns)

full_dataset = NephroDataset(static_df, ts_data, biopsy_df, phase='test')
#datapoint_limit = 1000
#dataset = Subset(full_dataset, range(datapoint_limit))
batch_size = 32
data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
static_scaler = full_dataset.scaler
cat_scaler = full_dataset.label_encoders
categorical_cardinalities = full_dataset.categorical_cardinalities
print(f"Categorical cardinalities: {categorical_cardinalities}")

checkpoint = torch.load('./model_checkpoints/models/best.ckpt', weights_only=False)
print(checkpoint.keys())
print(checkpoint['epoch'])

model = tame.AutoEncoder(categorical_cardinalities=categorical_cardinalities)
#Load weights
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

# load feature_mm file
with open('./feature_mm.json', 'r') as f:
    feature_mm = json.load(f)

# load feature_ms file
with open('./feature_ms.json', 'r') as f:
    feature_ms = json.load(f)


# After loading feature_mm
keys = list(feature_mm.keys())  # careful: assumes insertion order matches model

mins = torch.tensor([feature_mm[keys[i]][0] for i in range(15)], device=device)
maxs = torch.tensor([feature_mm[keys[i]][1] for i in range(15)], device=device)

means = torch.tensor([feature_ms[keys[i]][0] for i in range(15)], device=device)
stds = torch.tensor([feature_ms[keys[i]][1] for i in range(15)], device=device)

all_embeddings, all_ts_features, all_num_features, all_cat_features, static_vectors, all_pids, death_labels, graft_loss_labels, \
    graft_loss_days, mort_days, rej_days = extract_features_and_labels(model, data_loader)

patient_ids, dist_matrix = compute_similarity_matrix(all_pids, all_ts_features)
pids, emb_dist_matrix = compute_similarity_matrix(all_pids, all_embeddings)

# Make sure dist_results directory exists
os.makedirs("./dist_results", exist_ok=True)

# save dis matrix into a file 
np.save('./dist_results/dist_matrix.npy', dist_matrix)
np.save('./dist_results/emb_dist_matrix.npy', emb_dist_matrix)
# save patient ids into a file#
np.save('./dist_results/patient_ids.npy', patient_ids)
np.save('./dist_results/emb_patient_ids.npy', pids)
