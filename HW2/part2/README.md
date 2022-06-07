# Image Classification

Train myResnet by using residual block to CNN model and score 78.8% on the prediction of public dataset from Cifar-10.

- - -

### Usage 

Run `main.py` to generate model params in `./save_dir/Resnet/best_model.pt`. Training log file can be found in `./acc_log/Resnet/acc_Resnet_.log`

```shell
python3 main.py
```

Run `eval.py` to utilize `best_model.pt` to evaluate on public/private dataset

```shell
python3 eval.py
```
---

### Directory layout

    .
    └── part2 
       ├── cfg.py              # model configs
       ├── eval.py             # evaluation         
       ├── main.py             # main program to run
       ├── tool.py             # functions used in training
       ├── myDatasets.py       # generate dataset
       ├── myModels.py         # myLeNet and myResnet
       └── p2_data
           ├── annotations       # lables of test and train imgs
           ├── public_test       # 5000 of 32x32 test imgs
           ├── private_test      # directory to store private dataset
           ├── train             # 23000 of 32x32 labled imgs
           └── unlabled          # 30000 of 32x32 unlabled imgs