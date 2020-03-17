import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class HAM(nn.Module):
    def __init__(self, num_users, num_items, model_args, device):
        super(HAM, self).__init__()

        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d

        #self.V = 16

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims).to(device)
        self.item_embeddings = nn.Embedding(num_items, dims).to(device)

        #self.WS1 = nn.Linear(dims, self.V)
        #self.WS2 = nn.Linear(self.V, 3)

        self.W2 = nn.Embedding(num_items, dims, padding_idx=0).to(device)
        self.b2 = nn.Embedding(num_items, 1, padding_idx=0).to(device)

        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, item_seq, user_ids, items_to_predict, for_pred=False):
        item_embs = self.item_embeddings(item_seq)
        #low-order is fixed to 2
        short_item_embs = item_embs[:, -2:]

        user_emb = self.user_embeddings(user_ids)

        if not for_pred:
            item_embs = F.dropout(item_embs)

        union_out = item_embs.mean(1)

        w2 = self.W2(items_to_predict)
        b2 = self.b2(items_to_predict)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()

            # MF
            res = user_emb.mm(w2.t()) + b2

            # high-order
            res += union_out.mm(w2.t())

            # low-order
            rel_score = torch.matmul(short_item_embs, w2.t().unsqueeze(0))
            rel_score = torch.mean(rel_score, dim=1)
            
            res += rel_score
        else:
            # MF
            res = torch.baddbmm(b2, w2, user_emb.unsqueeze(2)).squeeze()

            # high-order
            res += torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

            # low-order
            rel_score = short_item_embs.bmm(w2.permute(0, 2, 1))
            rel_score = torch.mean(rel_score, dim=1)

            res += rel_score

        return res
