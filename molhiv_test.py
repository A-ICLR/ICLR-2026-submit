from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader,Data
import torch_geometric as pyg
dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')
split_idx = dataset.get_idx_split() 
ogbg_molhiv_pg_list = [graph for graph in dataset[split_idx["train"]]]
train_loader = DataLoader(ogbg_molhiv_pg_list, batch_size=32, shuffle=True)
print(ogbg_molhiv_pg_list[0])
edge_dim=ogbg_molhiv_pg_list[0].edge_attr.shape[1]
ogbg_molhiv_pg_list = [graph for graph in dataset[split_idx["valid"]]]
valid_loader = DataLoader(ogbg_molhiv_pg_list, batch_size=32, shuffle=False)
ogbg_molhiv_pg_list = [graph for graph in dataset[split_idx["test"]]]
test_loader = DataLoader(ogbg_molhiv_pg_list, batch_size=32, shuffle=False)
from ogb.graphproppred import Evaluator
from models import TransformerConv, DiffAttConv, DGATv2Conv
import argparse


# You can learn the input and output format specification of the evaluator as follows.
# print(evaluator.expected_input_format) 
# print(evaluator.expected_output_format) 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
import tqdm 

#torch.autograd.set_detect_anomaly(True)


class GNN(nn.Module):
      def __init__(self,in_dim,h_dim,out_dim,non_linearity,edge_dim=None,drop=0.3,layers=2,module="GAT",num_heads=1,gram=True,w_gate=False,t_gate=False):
          super(GNN, self).__init__()
          self.layers=layers
          self.conv=torch.nn.ModuleList()
          if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATv2": 
                  self.conv.append(pyg.nn.GATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(TransformerConv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,beta=True))
          elif module == "DGAT": 
                  self.conv.append(DGATv2Conv(in_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,w_gate=w_gate,t_gate=t_gate))
          else:
                  self.conv.append(DiffAttConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,gram=gram,w_gate=w_gate,t_gate=t_gate))
          for t in range(layers-2):
              if module == "GAT":
                  self.conv.append(pyg.nn.GATConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "GATed":
                  self.conv.append(pyg.nn.GATv2Conv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
              elif module == "Transf":
                  self.conv.append(TransformerConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,beta=True))
              elif module == "DGAT":
                  self.conv.append(DGATv2Conv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,w_gate=w_gate,t_gate=t_gate))
              else: 
                  self.conv.append(DiffAttConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,gram=gram,w_gate=w_gate,t_gate=t_gate))
              self.batchf=torch.nn.BatchNorm1d(out_dim)
          if module == "GAT":
             self.conv.append(pyg.nn.GATConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "GATed":
             self.conv.append(pyg.nn.GATv2Conv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False))
          elif module == "Transf":
                  self.conv.append(TransformerConv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False,beta=True))
          elif module == "DGAT": 
             self.conv.append(DGATv2Conv(h_dim,out_dim,edge_dim=edge_dim,heads=num_heads,concat=False,w_gate=w_gate,t_gate=t_gate))
          else:
             self.conv.append(DiffAttConv(h_dim,h_dim,edge_dim=edge_dim,heads=num_heads,concat=False,gram=gram,w_gate=w_gate,t_gate=t_gate))
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

@torch.no_grad()
def test(model, device):
    evaluator = Evaluator(name = "ogbg-molhiv") 
    model.eval()
    #decoder.eval()
    out_list=[]
    y_true_test=[]
    for graph in dataset[split_idx["test"]]:
        if batch.x.shape[0] == 1:
            pass
        else:
           out = model(graph.x.type(torch.FloatTensor).to(device), graph.edge_index.to(int).to(device),graph.edge_attr.type(torch.FloatTensor).to(device))
        #out_list.append(decoder(out).max(dim=0).values)
           out_list.append(out.max(dim=0)[0].mean(dim=0,keepdim=True))
           y_true_test.append(graph.y)
    #print(out.size())
    out_list_val=[]
    y_true_val=[]
    for graph in dataset[split_idx["valid"]]:
        if graph.x.shape[0] == 1:
            pass
        else:
           out = model(graph.x.type(torch.FloatTensor).to(device), graph.edge_index.to(int).to(device),graph.edge_attr.type(torch.FloatTensor).to(device))
           out_list_val.append(out.max(dim=0)[0].mean(dim=0,keepdim=True))
           y_true_val.append(graph.y)

    y_true_test = torch.cat(y_true_test,dim=0).cpu()
    y_true_val = torch.cat(y_true_val,dim=0).cpu()
    y_pred_test = torch.cat(out_list).cpu()
    y_pred_val = torch.cat(out_list_val).cpu()
    val_acc = evaluator.eval({
        'y_true': y_true_val.squeeze().unsqueeze(1),
        'y_pred': y_pred_val.squeeze().unsqueeze(1),
    })["rocauc"]
    test_acc = evaluator.eval({
        'y_true': y_true_test.squeeze().unsqueeze(1),
        'y_pred': y_pred_test.squeeze().unsqueeze(1),
    })["rocauc"]
    return val_acc, test_acc         
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("--------------------------------------------------GAT-------------------------------------------------------------")
model = GNN(dataset[0].x.shape[1], 256, dataset.num_classes,torch.nn.LeakyReLU(),edge_dim=edge_dim, layers=2,num_heads=8)
model=model.to(device)
print(next(model.parameters()).device)
#decoder=KanDecoder(dataset.num_classes)
#decoder=decoder.to(device)
epochs = 100
optimizer = torch.optim.Adam(list(model.parameters()),lr=0.03) #+list(decoder.parameters()), lr=0.03)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
for epoch in tqdm.tqdm(range(1, epochs),total=epochs):
    
    model.train()
    #decoder.train()

    total_loss = total_correct = 0
    i=0
    
    for batch in train_loader:
        batch_size = batch.batch_size
        optimizer.zero_grad()
        out_list=[]
        batch_y=[]
        for ba in range(0,len(batch)):
          out = model(batch[ba].x.type(torch.FloatTensor).to(device), batch[ba].edge_index.to(int).to(device),batch[ba].edge_attr.type(torch.FloatTensor).to(device))
          #out = decoder(out)
          out_list.append(out.max(dim=0)[0].mean(dim=0,keepdim=True))
          #print(out.mean(dim=0).argmax(dim=0))
          #print(batch[ba].y)
          batch_y.append(batch[ba].y)
        #print(out_list)
        out= torch.cat(out_list)
        batch_y=torch.cat(batch_y,dim=0)
        batch_y = torch.reshape(batch_y, (-1,)).to(device)
        #print(len(batch_y))
        loss = F.binary_cross_entropy_with_logits(out.type(torch.FloatTensor).unsqueeze(1).to(device), batch_y.type(torch.FloatTensor).unsqueeze(1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
        i+=1
    #pbar.close()
    loss = total_loss / len(split_idx["train"])
    approx_acc = total_correct / split_idx["train"].size(0)

    val_acc, test_acc = test(model, device)
    
    scheduler.step()
    
    print(f'Epoch {epoch:02d} |  Val: {val_acc:.4f}, Test: {test_acc:.4f}')

print("--------------------------------------------------GATv2-----------------------------------------------------------")
model = GNN(dataset[0].x.shape[1], 256, dataset.num_classes,torch.nn.LeakyReLU(), edge_dim=edge_dim, layers=2,module="GATv2",num_heads=8)
model=model.to(device)
#decoder=KanDecoder(dataset.num_classes)
#decoder=decoder.to(device)
epochs = 100
optimizer = torch.optim.Adam(list(model.parameters()) , lr=0.03) #+list(decoder.parameters()), lr=0.03)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
for epoch in tqdm.tqdm(range(1, epochs),total=epochs):
    
    model.train()
    #decoder.train()

    total_loss = total_correct = 0
    i=0
    
    for batch in train_loader:
        batch_size = batch.batch_size
        optimizer.zero_grad()
        out_list=[]
        batch_y=[]
        for ba in range(0,len(batch)):
          out = model(batch[ba].x.type(torch.FloatTensor).to(device), batch[ba].edge_index.to(int).to(device),batch[ba].edge_attr.type(torch.FloatTensor).to(device))
          #out = decoder(out)
          out_list.append(out.mean(dim=0).mean(dim=0,keepdim=True))
          #print(out)
          batch_y.append(batch[ba].y)
        out= torch.cat(out_list)  
        batch_y=torch.cat(batch_y,dim=0)
        batch_y = torch.reshape(batch_y, (-1,)).to(device)
        loss = F.binary_cross_entropy_with_logits(out.type(torch.FloatTensor).unsqueeze(1).to(device), batch_y.type(torch.FloatTensor).unsqueeze(1).to(device))
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
        i+=1
    #pbar.close()
    loss = total_loss / len(split_idx["train"])
    approx_acc = total_correct / split_idx["train"].size(0)

    val_acc, test_acc = test(model, device)
    scheduler.step()
    print(f'Epoch {epoch:02d} |  Val: {val_acc:.4f}, Test: {test_acc:.4f}')

print("--------------------------------------------------DGAT-------------------------------------------------------------")
model = GNN(dataset[0].x.shape[1], 256, dataset.num_classes,torch.nn.LeakyReLU(), edge_dim=edge_dim, layers=2,module="DGAT",num_heads=2)
model=model.to(device)
torch.cuda.empty_cache()
#decoder=KanDecoder(dataset.num_classes)
#decoder=decoder.to(device)
epochs = 100
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.005) #+list(decoder.parameters()), lr=0.03)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
for epoch in tqdm.tqdm(range(1, epochs),total=epochs):

    model.train()
    #decoder.train()

    total_loss = total_correct = 0
    i=0

    for batch in train_loader:
        batch_size = batch.batch_size
        optimizer.zero_grad()
        out_list=[]
        batch_y=[]
        for ba in range(0,len(batch)):
          if batch[ba].x.shape[0] == 1:
            pass
          else:
            out = model(batch[ba].x.type(torch.FloatTensor).to(device), batch[ba].edge_index.to(int).to(device),batch[ba].edge_attr.type(torch.FloatTensor).to(device))
            #out = decoder(out)
            out_list.append(out.max(dim=0)[0].mean(dim=0,keepdim=True))
            #print(out)
            batch_y.append(batch[ba].y)
        out= torch.cat(out_list)
        batch_y=torch.cat(batch_y,dim=0)
        batch_y = torch.reshape(batch_y, (-1,)).to(device)
        loss = F.binary_cross_entropy_with_logits(out.type(torch.FloatTensor).unsqueeze(1).to(device), batch_y.type(torch.FloatTensor).unsqueeze(1).to(device))
        #loss.register_hook(lambda grad: print(grad))
        loss.backward()
        #print(loss.grad)
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
        i+=1
    #pbar.close()
    loss = total_loss / len(split_idx["train"])
    approx_acc = total_correct / split_idx["train"].size(0)
    #print(loss.grad)
    val_acc, test_acc = test(model, device)
    scheduler.step()
    print(f'Epoch {epoch:02d} |  Val: {val_acc:.4f}, Test: {test_acc:.4f}, Loss: {loss: .4f}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGBL-PPA (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--step_size', type=float, default=50)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--module', type=float, default="DGAT")
    args = parser.parse_args()

    print("--------------------------------------------------{module}-----------------------------------------------------------".format(module=args.module))
    model = GNN(dataset[0].x.shape[1], args.hidden_channels, dataset.num_classes,torch.nn.LeakyReLU(), edge_dim=edge_dim, layers=args.num_layers,module=args.module,num_heads=args.num_heads)
    model=model.to(device)
    #decoder=KanDecoder(dataset.num_classes)
    #decoder=decoder.to(device)
    epochs = 100
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr) #+list(decoder.parameters()), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in tqdm.tqdm(range(1, epochs),total=epochs):

        model.train()
        #decoder.train()

        total_loss = total_correct = 0
        i=0

        for batch in train_loader:
            batch_size = batch.batch_size
            optimizer.zero_grad()
            out_list=[]
            batch_y=[]
            for ba in range(0,len(batch)):
                if batch[ba].x.shape[0] == 1:
                    pass                                                                                                                                                                                       
                else: 
                    out = model(batch[ba].x.type(torch.FloatTensor).to(device), batch[ba].edge_index.to(int).to(device),batch[ba].edge_attr.type(torch.FloatTensor).to(device))
                    #out = decoder(out)
                    out_list.append(out.max(dim=0)[0].mean(dim=0,keepdim=True))
                    #print(out)
                    batch_y.append(batch[ba].y)
            out= torch.cat(out_list)
            batch_y=torch.cat(batch_y,dim=0)
            batch_y = torch.reshape(batch_y, (-1,)).to(device)
            loss = F.binary_cross_entropy_with_logits(out.type(torch.FloatTensor).unsqueeze(1).to(device), batch_y.type(torch.FloatTensor).unsqueeze(1).to(device))
            #loss.register_hook(lambda grad: print(grad))
            loss.backward()
            #print(loss.grad)
            optimizer.step()

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
            i+=1
        #pbar.close()
        loss = total_loss / len(split_idx["train"])
        approx_acc = total_correct / split_idx["train"].size(0)

        val_acc, test_acc = test(model, device)
        scheduler.step()
        print(f'Epoch {epoch:02d} |  Val: {val_acc:.4f}, Test: {test_acc:.4f}, Loss: {loss: .4f}')

