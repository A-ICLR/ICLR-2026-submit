import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm

import torch_geometric as pyg
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.utils import scatter
from models import DiffAttConv, DGATv2Conv

dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)
edge_dim= data.edge_attr.shape[1]
# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True,
                                num_workers=5)
test_loader = RandomNodeLoader(data, num_parts=40, num_workers=5)

class GNN(nn.Module):
      def __init__(self,in_dim,h_dim,out_dim,non_linearity,edge_dim=None,drop=0.3,layers=2,module="GAT",num_heads=1, real_att=False):
          super(GNN, self).__init__()
          self.layers=layers
          self.conv=torch.nn.ModuleList()
          if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATv2":
                  self.conv.append(pyg.nn.GATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))        
          elif module == "DTrans": 
                  self.conv.append(DiffAttConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(pyg.nn.TransformerConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          else: 
                  self.conv.append(DGATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False, temperature=False,gate_w_orth=True))
          for t in range(layers-2):
              if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "GATv2":
                  self.conv.append(pyg.nn.GATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))    
              elif module == "DTrans":
                  self.conv.append(DiffAttConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "Transf":
                  self.conv.append(pyg.nn.TransformerConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              else:
                  self.conv.append(DGATv2Conv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False, gate_w_orth= True))
              self.batchf=torch.nn.BatchNorm1d(out_dim)
          if module == "GAT":
             self.conv.append(pyg.nn.GATConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATv2":
                  self.conv.append(pyg.nn.GATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))   
          elif module == "DTrans":
             self.conv.append(DiffAttConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(pyg.nn.TransformerConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          else: 
             self.conv.append(DGATv2Conv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False, gate_w_orth=True))
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(data.x.shape[1], 256, data.y.size(-1),torch.nn.LeakyReLU(), edge_dim=edge_dim, layers=2,module="DGAT").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)
    scheduler.step()
    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGBN-PROTEINS (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--step_size', type=float, default=50)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--module', type=float, default="DGAT")
    args = parser.parse_args()

    model = GNN(data.x.shape[1], args.hidden_channels, data.y.size(-1),torch.nn.LeakyReLU(), edge_dim=edge_dim, layers=args.num_layers,module=args.module, num_heads=args.num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(1, 1001):
        #model = GNN(data.x.shape[1], 256, data.y.size(-1),torch.nn.LeakyReLU(), edge_dim=edge_dim, layers=2).to(device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)    
        loss = train(epoch)
        train_rocauc, valid_rocauc, test_rocauc = test()
        print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
            f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')

    print( "------------------------------------------------------------------GAT--------------------------------------------------------------------------------------------")
