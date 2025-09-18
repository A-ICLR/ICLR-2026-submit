import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data, Dataset
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm
import numpy as np
import torch_geometric as pyg
from models import DiffAttConv, DGATv2Conv
import argparse
import torch_geometric.transforms as T


class GNN(nn.Module):
      def __init__(self,in_dim,h_dim,out_dim,non_linearity,edge_dim=None,drop=0.2,layers=2,module="GAT",num_heads=1):
          super(GNN, self).__init__()
          self.layers=layers
          self.conv=torch.nn.ModuleList()
          if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATv2":
                  self.conv.append(pyg.nn.GATv2Conv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))        
          elif module == "DTrans": 
                  self.conv.append(DiffAttConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(pyg.nn.TransformerConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          else: 
                  self.conv.append(DGATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,gate_w_orth=False))
          for t in range(layers-2):
              if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "GATv2":
                  self.conv.append(pyg.nn.GATv2Conv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))    
              elif module == "DTrans":
                  self.conv.append(DiffAttConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "Transf":
                  self.conv.append(pyg.nn.TransformerConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              else:
                  self.conv.append(DGATv2Conv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,gate_w_orth=False))
              self.batchf=torch.nn.BatchNorm1d(out_dim)
          if module == "GAT":
             self.conv.append(pyg.nn.GATConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATv2":
             self.conv.append(pyg.nn.GATv2Conv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "DTrans":
             self.conv.append(DiffAttConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(pyg.nn.TransformerConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          else: 
             self.conv.append(DGATv2Conv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False,gate_w_orth=False))
          self.batch=torch.nn.BatchNorm1d(h_dim)
          self.batchf=torch.nn.BatchNorm1d(out_dim)
          self.dropout=drop
          self.non_linearity=non_linearity      
      def forward(self,x,edge_index,edge_att=None):
          h=x
          for y in range(self.layers-1):
              h=self.conv[y](h,edge_index,edge_att)
              h=self.batch(h)
              h=self.non_linearity(h)
              h=F.dropout(h,p=self.dropout,training=self.training)
          h=self.conv[self.layers-1](h,edge_index,edge_att)
          h=self.batchf(h)
          #h=self.non_linearity(h)
          return h




class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, x, adj_t, split_edge, optimizer, batch_size):

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=64* 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--module', type=int, default="DGAT")
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    model = GNN(args.hidden_channels, args.hidden_channels,
                     args.hidden_channels,torch.nn.LeakyReLU(), layers=args.num_layers,
                     module=args.module,num_heads=args.num_heads).to(device)

    emb = torch.nn.Embedding(data.adj_t.size(0),
                             args.hidden_channels).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              1, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')

    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        #model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, emb.weight, adj_t, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, emb.weight, adj_t, split_edge,
                               evaluator, args.batch_size)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')


if __name__ == "__main__":
    main()
