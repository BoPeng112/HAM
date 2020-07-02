import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class Model(nn.Module):
    def __init__(self, num_users, num_items, model_args, device):
        super(Model, self).__init__()

        self.args = model_args

        # init args
        L = self.args.L
        P = self.args.P
        dims = self.args.d

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims).to(device)
        self.item_embeddings = nn.Embedding(num_items, dims).to(device)

        self.decay = torch.tensor([[1.0]*(L-P) + [2]*P]).view(-1,1).to(device)

        self.W2 = nn.Embedding(num_items, dims, padding_idx=0).to(device)
        self.b2 = nn.Embedding(num_items, 1, padding_idx=0).to(device)

        self.cor_gate = nn.Linear(dims, 1)
        self.mean_gate = nn.Linear(dims,1)

        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, item_seq, user_ids, items_to_predict, for_pred=False):
        item_embs = self.item_embeddings(item_seq)
        user_emb = self.user_embeddings(user_ids)
        if not for_pred:
            item_embs = F.dropout(item_embs)

        short_item_embs = item_embs[:, -self.args.P:]
        low_order       = short_item_embs.mean(1)
        high_order      = item_embs.mean(1)
        union_out   = low_order + high_order

        second_out  = torch.mul(item_embs.unsqueeze(2), item_embs.unsqueeze(1)).sum(2)
        third_out   = torch.mul(second_out.unsqueeze(2), item_embs.unsqueeze(1)).sum(2)
        fourt_out   = torch.mul(third_out.unsqueeze(2), item_embs.unsqueeze(1)).sum(2)
        
        if self.args.order == 2:
            union_out = union_out + second_out.mean(1) * high_order
        elif self.args.order == 3:
            union_out = union_out + (second_out.mean(1) + third_out.mean(1)) * high_order
        elif self.args.order == 4:
            union_out = union_out + (second_out.mean(1) + third_out.mean(1) + fourt_out.mean(1)) * high_order

        w2 = self.W2(items_to_predict)
        b2 = self.b2(items_to_predict)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()

            #user-item
            res = user_emb.mm(w2.t()) + b2
            #item associations
            res += union_out.mm(w2.t())

        else:
            #user-item
            res = torch.baddbmm(b2, w2, user_emb.unsqueeze(2)).squeeze()
            #item associations
            res += torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

        return res
