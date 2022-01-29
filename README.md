## Download dataset
```shell
bash download_dataset.sh {dataset_name} ./datasets
```
Here `{dataset_name}` is one of the datasets following:
- WaterDrop
- Water
- Sand
- Goop
- etc.

## Transform dataset into .pkl file
```shell
python transform_to_pkl.py --dataset {dataset_name} --split {split}
```
Here `{split}` is one of `train, test, valid`.

## Train a model.
```shell
python main.py --mode train --dataset {dataset}
```

## Evaluate.
- Evaluate one_step loss.
```shell
python main.py --mode eval --eval_split {split}
```
- Evaluate rollout loss.
```shell
python main.py --mode eval_rollout --eval_split {split}
```

## Render rollout.
```shell
python render_rollout.py --rollout_path rollouts/Water/gcn/rollout_test_0.pkl
```

If using `GAT` as our GNN backbone, we can get the following simulation effect on `Water`:

![results](results/rollout.gif)
