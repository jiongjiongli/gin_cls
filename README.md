# Graph Isomorphism Network Classification
## Dataset: 

- Name: MUTAG
- Split: Train: 135, Validation: 15, Test: 38

## Best Result:

- Validation Acc: 94.00% 
- Test Acc: 92.11%

<img width="865" height="710" alt="accuracy" src="https://github.com/user-attachments/assets/a66c9db8-0579-4a41-8566-05007a18319b" />

## Method: 

1. GINConv
2. Model hyperparameter selection: 10 Fold Cross Validation + Hyper-parameters Grid Search
3. Final step: Use all training data to train model.

## Reference:

- GINConv: https://arxiv.org/pdf/1810.00826
- Implementation: https://medium.com/@a.r.amouzad.m/how-to-get-state-of-the-art-result-on-graph-classification-with-gnn-73afadff5d49
