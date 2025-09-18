from models import GATedConv, DiffAttConv, TransformerConv, DGATConv
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from model import MPNNs

class GNN(nn.Module):
      def __init__(self,in_dim,h_dim,out_dim,non_linearity,edge_dim=None,drop=0.3,layers=2,module="GAT",num_heads=1):
          super(GNN, self).__init__()
          self.layers=layers
          self.conv=torch.nn.ModuleList()
          self.batch = torch.nn.ModuleList()
          self.dropout = drop
          if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATed": 
                  self.conv.append(GATedConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(TransformerConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,beta=True))
          else: 
                  self.conv.append(DiffAttConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          self.batch.append(torch.nn.BatchNorm1d(h_dim))
          for t in range(layers-2):
              if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "GATed":
                  self.conv.append(GATedConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "Transf":
                  self.conv.append(TransformerConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,beta=True))
              else:
                  self.conv.append(DiffAttConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              self.batch.append(torch.nn.BatchNorm1d(h_dim))
          if module == "GAT":
             self.conv.append(pyg.nn.GATConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATed":
             self.conv.append(GATedConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(TransformerConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False,beta=True))
          else: 
             self.conv.append(DiffAttConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          self.batch.append(torch.nn.BatchNorm1d(out_dim))
          self.non_linearity=non_linearity
          self.reset_parameters()
      def reset_parameters(self):
          for lin in self.batch:
              lin.reset_parameters()
          for lx in self.conv:
              lx.reset_parameters()      
      def forward(self,x,edge_index,edge_att=None):
          h=x
          for y in range(self.layers-1):
              h=self.conv[y](h,edge_index,edge_att)
              h=self.batch[y](h)
              h=self.non_linearity(h)
              h=F.dropout(h, p=self.dropout ,training=self.training)
          h=self.conv[self.layers-1](h,edge_index,edge_att)
          h=self.batch[-1](h)
          #h=self.non_linearity(h)
          return h

"""
def parse_method(args, n, c, d, device):
    
    model = GNN(d, args.hidden_channels, c, torch.nn.ReLU(), layers=args.local_layers, drop=args.dropout, 
    num_heads=args.num_heads, module=args.model).to(device)
    
    return model
"""        
def parse_method(args, n, c, d, device):
    
    model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout, 
    heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear, res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn = args.gnn).to(device)
    
    return model

def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', type=str, default='DGAT')
    # GNN
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    
    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')


