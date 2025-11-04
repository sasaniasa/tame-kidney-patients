import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import CONFIG

def value_embedding_data(d = 512, split = 101):
    vec = np.array([np.arange(split) * i for i in range(d // 2)], dtype=np.float32).transpose()
    vec = vec / vec.max() 
    embedding = np.concatenate((np.sin(vec), np.cos(vec)), 1)
    embedding[0, :d] = 0
    embedding = torch.from_numpy(embedding)
    assert not torch.isnan(embedding).any()
    return embedding
    

class StaticEncoder(nn.Module):
    def __init__(self, categorical_cardinalities):
        super(StaticEncoder, self).__init__()

        self.embedding_dim = 32
        self.output_dim = CONFIG['embedding_dim']

        # We assume categorical_cardinalities is a list like [5, 10, 3, ...]
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, self.embedding_dim)
            for cardinality in categorical_cardinalities
        ])

        self.numerical_dim = len(CONFIG['static_numerical_feat'])

        self.numerical_processor = nn.Sequential(
            nn.Linear(self.numerical_dim, self.embedding_dim),
            nn.ReLU()
        )

        total_input_dim = len(self.embeddings) * self.embedding_dim + self.embedding_dim 
        self.fc = nn.Linear(total_input_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, categorical_features: torch.Tensor, numerical_features: torch.Tensor):
        """
        categorical_features: Tensor[int], shape (B, num_categorical)
        numerical_features: Tensor[float], shape (B, num_numerical)
        """
        # Embed each categorical column
        embedded_cat = [emb(categorical_features[:, i].long()) for i, emb in enumerate(self.embeddings)]
        embedded_cat = torch.cat(embedded_cat, dim=-1)  # (B, C * embed_dim)

        # Process numerical features
        embedded_num = self.numerical_processor(numerical_features)  # (B, embed_dim)

        # Combine and project
        combined = torch.cat([embedded_cat, embedded_num], dim=-1)
        out = self.fc(combined)
        out = self.relu(out)

        return out


class AutoEncoder(nn.Module):
    def __init__(self, embed_size = CONFIG['embedding_dim'], hidden_size = CONFIG['hidden_dim'], categorical_cardinalities=None):
        super(AutoEncoder, self).__init__()
        #self.args = args
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.split_num = CONFIG['split_num']
        self.output_size = len(CONFIG['ts_feat'])
        #self.demo_embedding = FloatEmbedder(input_dim=12, embed_dim=512)
        #self.demo_cat_embedding = nn.Embedding(452, embed_size)


        # networks for value_order_embedding
        self.vocab_size = (len(CONFIG['ts_feat']) + 2) * (1 + self.split_num) + 5
        self.embedding = nn.Embedding(self.vocab_size, embed_size)

        self.value_embedding = nn.Embedding.from_pretrained(value_embedding_data(embed_size, self.split_num + 1))

        #self.time_value_embedding = nn.Embedding.from_pretrained(value_embedding_data(embed_size, CONFIG['max_rel_days'] + 1))

        self.static_encoder = StaticEncoder(categorical_cardinalities=categorical_cardinalities)

        self.value_mapping = nn.Sequential(
                nn.Linear ( embed_size * 2, embed_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                )

        
        self.mapping = nn.Sequential(
                nn.Linear ( 2 * embed_size, embed_size),
                nn.ReLU ( ),
                nn.Linear ( embed_size, embed_size),
                nn.ReLU ( ),
                )
        
        self.lstm = nn.LSTM (input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        
        self.output = nn.Sequential (
            nn.Linear (2*hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.2),
            nn.Linear ( hidden_size, embed_size),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.pre_embedding = nn.Sequential(
                nn.Linear (embed_size * 2, embed_size),
                nn.ReLU ( ),
                nn.Linear ( embed_size, embed_size),
                nn.ReLU ( ),
                )
        self.post_embedding = nn.Sequential(
                nn.Linear ( embed_size * 2, embed_size),
                nn.ReLU ( ),
                nn.Linear ( embed_size, embed_size),
                nn.ReLU ( ),
                )

        self.tah_mapping = nn.Sequential (
                nn.Linear(2*hidden_size, hidden_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                nn.Linear ( hidden_size, embed_size),
        ) 
        self.tav_mapping = nn.Sequential (
                nn.Linear(embed_size, embed_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                nn.Linear ( embed_size, embed_size),
        )
        self.value = nn.Sequential (
            nn.Linear (embed_size, hidden_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.2),
            nn.Linear (hidden_size, self.output_size)
        )

        self.layer_norm = nn.LayerNorm(embed_size)
        self.mapping_norm = nn.LayerNorm(embed_size)


    def visit_pooling(self, x):
        output = x
        size = output.size()
        output = output.view(size[0] * size[1], size[2], output.size(3))    # (64*30, 13, 512)
        output = torch.transpose(output, 1,2).contiguous()                  # (64*30, 512, 13)
        output = self.pooling(output)                                       # (64*30, 512, 1)
        output = output.view(size[0], size[1], size[3])                     # (64, 30, 512)
        return output
        


    def value_order_embedding(self, x):
        size = list(x[0].size())               # (64, 30, 13)
        index, value = x
        xi = self.embedding(index.view(-1))          # (64*30*13, 512)
        # xi = xi * (value.view(-1).float() + 1.0 / self.args.split_num)
        #v = value.view(-1)
        #print("forward: max index for value_embedding:", v.max().item(), "min:", v.min().item())
        #print("value_embedding size:", self.value_embedding.num_embeddings)
        xv = self.value_embedding(value.view(-1))    # (64*30*13, 512)
        x = torch.cat((xi, xv), 1)                   # (64*30*13, 1024)
        x = self.value_mapping(x)                 # (64*30*13, 512)   
        size.append(-1)
        x = x.view(size)                    # (64, 30, 13, 512)
        return x



    def pp_value_embedding(self, neib):
        size = list(neib[1].size())
        # print(type(neib[0]))
        # print(len(neib[0]))

        pre_x = self.value_order_embedding(neib[0])
        post_x = self.value_order_embedding(neib[2])

        pre_t = self.value_embedding(neib[1].view(-1))
        post_t = self.value_embedding(neib[3].view(-1))

        size.append(-1)
        pre_t = pre_t.view(size)
        post_t = post_t.view(size)

        pre_x = self.pre_embedding(torch.cat((pre_x, pre_t), 3))
        post_x = self.post_embedding(torch.cat((post_x, post_t), 3))
        return pre_x, post_x
    
    
    def time_aware_attention(self, hidden, vectors, mask=None):
        # hidden [2, 1826, 256]
        # vectors [2, 1826, 30, 512]
        #print(hidden.shape) # lstm_out 
        #print(vectors.shape) # pp
        wh = self.tah_mapping(hidden)
        wh = wh.unsqueeze(2)
        #print(wh.shape)# torch.Size([2, 1826, 1, 256])
        #print(vectors.shape) # torch.Size([2, 1826, 30, 512])
        wh = wh.expand_as(vectors)
        #print(wh.shape)#torch.Size([2, 1826, 30, 512])
        wv = self.tav_mapping(vectors)
        beta = wh + wv # torch.Size([2, 1826, 30, 512])

        if mask is not None:
            attention_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(beta)
            beta = beta.masked_fill(attention_mask == 0, float('-inf'))
            # That would make softmax undefined (NaN), so we replace them with zeros
            all_inf_mask = torch.isinf(beta).all(dim=2, keepdim=True)  # [B, T, 1, D]
            beta = beta.masked_fill(all_inf_mask, 0.0)

        #print(beta.shape)
        alpha = F.softmax(beta, 2)

        alpha_trans = alpha.transpose(2,3).contiguous()
        vectors = vectors.transpose(2,3).contiguous()
        vsize = list(vectors.size()) # [64, 30, 512, 54]

        alpha_trans = alpha_trans.view((-1, 1, alpha_trans.size(3)))
        vectors = vectors.view((-1, vectors.size(3), 1))
        # print(alpha.size())
        # print(vectors.size())
        att_res = torch.bmm(alpha_trans, vectors)
        # print(att_res.size())
        att_res = att_res.view(vsize[:3])
        # print(att_res.size())
        # return att_res
        return att_res, alpha

    
    def forward(self, ts_data, neib, numerical_features, categorical_features, value_mask=None, mask=None, get_embedding=False, plot_att=False):
        
        #print('ts_data', ts_data.shape) # ([1, 1745, 15])
        ts_data = self.value_order_embedding(ts_data) # [B, T, F, E]
        #print(ts_data.shape) # torch.Size([1, 1745, 15, 512])

        demo = self.static_encoder(categorical_features, numerical_features) # [B, E]
        #print(demo.shape) # torch.Size([1, 512])
        
        # demo: [B, E] → expand to [B, T, F, E]
        demo = demo.unsqueeze(1).unsqueeze(2).expand(-1, ts_data.size(1), ts_data.size(2), -1)
        #print(demo.shape) # torch.Size([1, 1745, 15, 512])

        x = torch.cat((ts_data, demo), 3)
        x = self.mapping(x) 

        x = self.mapping_norm(x) # [B, T, F, E]

        #print(x.shape) # torch.Size([1, 1745, 15, 512])

        x = self.visit_pooling(x)
        #print(x.shape) # torch.Size([1, 1745, 512])

        # lstm
        lstm_out, _ = self.lstm( x )  # torch.Size([1, 1745, 512])

        if get_embedding:
            return lstm_out
        
        out = self.output(lstm_out) # torch.Size([1, 1745, 512])

        # layer normalization
        out = self.layer_norm(out) # torch.Size([1, 1745, 512])

        #print('lstm_out', lstm_out.shape) 

        pre_x, post_x = self.pp_value_embedding(neib)
        pp = torch.cat((pre_x, post_x), 2) 

        att_res, alpha = self.time_aware_attention(lstm_out, pp, mask)

        if plot_att:
            return alpha

        out = out + att_res

        #print(out.shape) # ([1, 1745, 512])

        #if mask is not None:
        #    out = out * mask.unsqueeze(-1).float()

        out = self.value(out)

        #print(out.shape) # torch.Size([B, 473, 15])

        return out
