# HLN-DDI
A hierarchical learning network for drug-drug interactions prediction with molecular structure and co-attention mechanism

## Requirements
``` conda env create -f environment.yml```

# Reproduction
## Clone this project
```bash
git clone https://github.com/likeeMoon/HLN-DDI.git
cd HLN-DDI
```


## Train model and make prediction under transductive setting
Run by command line, e.g.:
`python ddi_main.py --dataset DrugBank `

## Train model and make prediction under inductive setting
Run by command line, e.g.:
`python ddi_main.py --dataset DrugBank  --inductive`

For detailed command line options, see `ddi_main.py`
