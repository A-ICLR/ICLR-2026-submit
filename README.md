# Usage & Experiment Reproduction

## Installation 

Install the required dependencies:

```bash
pip install torch
pip install torch_geometric
pip install ogb
pip install networkx
pip install numpy
pip install tqdm
```

## Models

The following custom layers are implemented in models.py:

* DGATConv (Difference Graph Attention)

* DiffAttConv (Difference Transformer)

Both are automatically loaded through the GNN wrapper inside each dataset script.
All models are implemented using PyTorch Geometric.

## Running Experiments
### Configurable Parameters

Each test script supports the following parameters:

* **num_heads**: int — number of attention heads

* **lr**: float — learning rate

* **module**: ["DGAT", "DTransf", "GATv2", "GAT", "Transf"] — model selection

* **dropout**: float — dropout rate

* **gamma**: float — scheduler gamma factor

* **step_size**: int — scheduler step size

* **layers**: int — number of layers
#### Homophilic Datasets

For each dataset, run the corresponding script:

###### MolHIV:
&thinsp;
```bash
python molhiv_test.py
```

###### Proteins:
&thinsp;
```bash
python proteins_test.py
```


###### PPA:
&thinsp;
```bash
python ppa_test.py
```

###### DDI:
&thinsp;
```bash
python ddi_test.py
```





##### Example Commands

###### MolHIV
&thinsp;
```bash
python molhiv_test.py --module DGAT --num_heads 4 --lr 0.001 --dropout 0.5 --gamma 0.5 --step_size 50
python molhiv_test.py --module DTransf --num_heads 4 --lr 0.001 --dropout 0.5
python molhiv_test.py --module GAT --num_heads 8 --lr 0.001 --dropout 0.5
python molhiv_test.py --module GATv2 --num_heads 8 --lr 0.001 --dropout 0.5
python molhiv_test.py --module Transf --num_heads 8 --lr 0.001 --dropout 0.5
```

###### Proteins
&thinsp;
```bash
python proteins_test.py --module DGAT --num_heads 1 --lr 0.005 --dropout 0.3
python proteins_test.py --module DTransf --num_heads 1 --lr 0.005 --dropout 0.3
python proteins_test.py --module GAT --num_heads 2 --lr 0.005 --dropout 0.3
python proteins_test.py --module GATv2 --num_heads 2 --lr 0.005 --dropout 0.3
python proteins_test.py --module Transf --num_heads 2 --lr 0.005 --dropout 0.3
```

###### PPA
&thinsp;
```bash
python ppa_test.py --module DGAT --num_heads 1 --lr 0.005 --dropout 0.1
python ppa_test.py --module DTransf --num_heads 1 --lr 0.005 --dropout 0.1
python ppa_test.py --module GAT --num_heads 2 --lr 0.005 --dropout 0.3
python ppa_test.py --module GATv2 --num_heads 2 --lr 0.005 --dropout 0.3
python ppa_test.py --module Transf --num_heads 2 --lr 0.005 --dropout 0.3
```

###### DDI
&thinsp;
```bash
python ddi_test.py --module DGAT --num_heads 1 --lr 0.001 --dropout 0.2
python ddi_test.py --module DTransf --num_heads 1 --lr 0.001 --dropout 0.2
python ddi_test.py --module GAT --num_heads 2 --lr 0.001 --dropout 0.5
python ddi_test.py --module GATv2 --num_heads 2 --lr 0.001 --dropout 0.5
python ddi_test.py --module Transf --num_heads 2 --lr 0.001 --dropout 0.5
```

#### Heterophilic Datasets

For heterophilic benchmarks (Minesweeper, Roman Empire, Amazon-Ratings, Questions), use:

```bash
cd heterophilic_test/medium_graph
python main.py --dataset roman-empire
```


--dataset can be set to minesweeper, questions, amazon-ratings, or roman-empire.

By default, it is set to roman-empire.

##### Example Commands
###### Roman-Empire
&thinsp;
```bash
python main.py --dataset roman-empire --module DGAT --num_heads 2 --lr 0.005 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset roman-empire --module DTransf --num_heads 2 --lr 0.005 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset roman-empire --module GAT --num_heads 4 --lr 0.005 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset roman-empire --module GATv2 --num_heads 4 --lr 0.005 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset roman-empire --module Transf --num_heads 4 --lr 0.005 --dropout 0.3 --res --jk --bn --pre_linear
```
###### Minesweeper
&thinsp;
```bash
python main.py --dataset minesweeper --module DGAT --num_heads 3 --lr 0.001 --dropout 0.2 --res --jk --bn --pre_linear
python main.py --dataset minesweeper --module DTransf --num_heads 3 --lr 0.001 --dropout 0.2 --res --jk --bn --pre_linear
python main.py --dataset minesweeper --module GAT --num_heads 4 --lr 0.001 --dropout 0.2 --res --jk --bn --pre_linear
python main.py --dataset minesweeper --module GATv2 --num_heads 4 --lr 0.001 --dropout 0.2 --res --jk --bn --pre_linear
python main.py --dataset minesweeper --module Transf --num_heads 4 --lr 0.001 --dropout 0.2 --res --jk --bn --pre_linear
```
###### Amazon-Ratings
&thinsp;
```bash
python main.py --dataset amazon-ratings --module DGAT --num_heads 1 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset amazon-ratings --module DTransf --num_heads 1 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset amazon-ratings --module GAT --num_heads 2 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset amazon-ratings --module GATv2 --num_heads 2 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset amazon-ratings --module Transf --num_heads 2 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
```
###### Questions
&thinsp;
```bash
python main.py --dataset questions --module DGAT --num_heads 1 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset questions --module DTransf --num_heads 1 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset questions --module GAT --num_heads 2 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset questions --module GATv2 --num_heads 2 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
python main.py --dataset questions --module Transf --num_heads 2 --lr 0.001 --dropout 0.3 --res --jk --bn --pre_linear
```

### Experiment Protocol

All experiments (homophilic + heterophilic) were run 10 times per configuration to avoid parameter contamination and ensure statistical reliability.

##### In Heterophilic Tests

We used ResNet, Batch Normalization, and Jumping Knowledge (JK) in every model and baseline for fairness.
Example execution scripts are available inside each test directory.

#### Acknowledgements

The training loops for Minesweeper, Amazon-Ratings, Questions, and Roman Empire were adapted from publicly available implementations of Classic GNNs.
The model implementations have been replaced with our own code.