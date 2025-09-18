import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data, Dataset
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm
import numpy as np
import torch_geometric as pyg
import argparse
from models import DiffAttConv, DGATv2Conv


class GNN(nn.Module):
      def __init__(self,in_dim,h_dim,out_dim,non_linearity,edge_dim=None,drop=0.1,layers=2,module="GAT",num_heads=1):
          super(GNN, self).__init__()
          self.layers=layers
          self.conv=torch.nn.ModuleList()
          if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "DTrans": 
                  self.conv.append(DiffAttConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(pyg.nn.TransformerConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATv2":
                  self.conv.append(pyg.nn.GATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))          
          else: 
                  self.conv.append(DGATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          for t in range(layers-2):
              if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "DTrans":
                  self.conv.append(DiffAttConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "Transf":
                  self.conv.append(pyg.nn.TransformerConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "GATv2":
                  self.conv.append(pyg.nn.GATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))   
              else:
                  self.conv.append(DGATv2Conv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              self.batchf=torch.nn.BatchNorm1d(out_dim)
          if module == "GAT":
             self.conv.append(pyg.nn.GATConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "DTrans":
             self.conv.append(DiffAttConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(pyg.nn.TransformerConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATv2":
                  self.conv.append(pyg.nn.GATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))       
          else: 
             self.conv.append(DGATv2Conv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
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

import torch
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

class PPAModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(out_dim*2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index):
        h = self.encoder(x)
        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]
        return torch.sigmoid(self.pred(torch.cat([h_src, h_dst], dim=-1)))

def train(model, device, data, split_edge, optimizer, batch_size=65536):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    # Prepare edges
    pos_edge_index = split_edge['train']['edge'].t().to(device)
    
    total_loss = 0
    for start in range(0, pos_edge_index.size(1), batch_size):
        end = min(start + batch_size, pos_edge_index.size(1))
        
        # Process current batch
        pos_batch = pos_edge_index[:, start:end]
        neg_batch = negative_sampling(
            edge_index=pos_batch,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_batch.size(1)
        ).to(device)
        
        optimizer.zero_grad()
        
        pos_pred = model(data.x.type(torch.float), pos_batch)
        h_src = pos_pred[pos_batch[0]]  # [num_edges, out_dim]
        h_dst = pos_pred[pos_batch[1]]  # [num_edges, out_dim]
        
        # Compute dot product similarity
        pos_logits = torch.sum(h_src * h_dst, dim=-1)
        neg_pred = model(data.x.type(torch.float), neg_batch)
        h_src = neg_pred[neg_batch[0]]  # [num_edges, out_dim]
        h_dst = neg_pred[neg_batch[1]]  # [num_edges, out_dim]
        
        neg_logits = torch.sum(h_src * h_dst, dim=-1)
        pos_target = torch.ones_like(pos_logits)  # [batch_size]
        neg_target = torch.zeros_like(neg_logits)  # [batch_size]
        
        # Compute loss
        loss = loss_fn(pos_logits, pos_target) + loss_fn(neg_logits, neg_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (pos_edge_index.size(1) // batch_size + 1)

@torch.no_grad()
def evaluate(model, device, data, edges, evaluator, batch_size=131072):
    model.eval()
    
    pos_edge_index = edges['edge'].t().to(device)
    neg_edge_index = edges['edge_neg'].t().to(device)
    
    # Process positive edges
    pos_preds = []
    for start in range(0, pos_edge_index.size(1), batch_size):
        end = min(start + batch_size, pos_edge_index.size(1))
        pos = model(data.x.type(torch.float), pos_edge_index[:, start:end])
        h_src = pos[pos_edge_index[0, start:end]]  # [num_edges, out_dim]
        h_dst = pos[pos_edge_index[1, start:end]]  # [num_edges, out_dim]
        pos = torch.sum(h_src * h_dst, dim=-1).cpu()
        pos_preds.append(pos)
    
    # Process negative edges
    neg_preds = []
    for start in range(0, neg_edge_index.size(1), batch_size):
        end = min(start + batch_size, neg_edge_index.size(1))
        neg = model(data.x.type(torch.float), neg_edge_index[:, start:end])
        h_src = neg[neg_edge_index[0, start:end]]  # [num_edges, out_dim]
        h_dst = neg[neg_edge_index[1, start:end]]  # [num_edges, out_dim]
        pos = torch.sum(h_src * h_dst, dim=-1).cpu()
        neg_preds.append(neg)
    
    return evaluator.eval({
        'y_pred_pos': torch.cat(pos_preds).squeeze(),
        'y_pred_neg': torch.cat(neg_preds).squeeze()
    })

def main():
    parser = argparse.ArgumentParser(description='OGBL-PPA (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--step_size', type=float, default=50)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--module', type=float, default="DGAT")
    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = PygLinkPropPredDataset(name='ogbl-ppa')
    data = dataset[0].to(device)
    split_edge = dataset.get_edge_split()
    
    # Initialize model
    model = GNN(data.num_features, args.hidden_channels, 1 ,torch.nn.LeakyReLU(), layers=args.num_layers,module=args.module, num_heads=args.num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    evaluator = Evaluator(name='ogbl-ppa')
    
    # Training loop
    best_hits = 0
    for epoch in range(1, 101):
        loss = train(model, device, data, split_edge, optimizer)
        valid_result = evaluate(model, device, data, split_edge['valid'], evaluator)
        hits = valid_result['hits@100']

        test_result = evaluate(model, device, data, split_edge['test'], evaluator)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Hits@100_val: {hits:.4f}, Hits@100_test: {test_result["hits@100"]:.4f}')
        scheduler.step()

if __name__ == '__main__':
    main()
