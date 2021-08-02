# RGNN
This is the implementation of the following paper:
> Yong Liu, Susen Yang, Yinan Zhang, Chunyan Miao, Zaiqing Nie, and Juyong Zhang. "Learning Hierarchical Review Graph Representations for Recommendation." IEEE Transactions on Knowledge and Data Engineering (2021).

## Environment Requirement
The code has been tested running under Python 3.5. The required packages are as follows:
* pytorch
* pytorch-geometric
* nltk
* numpy
* scipy
* networkx

## Example to Run the Codes
The instruction of commands can be found in the source codes (see main function in model/train.py).
* python train.py
* python train.py --dataset music --batch_size 128 --num_layers 2 --dim 16 --word_dim 16 --hidd_dim 8 --factors 8 --lr 0.005 --l2_re 0.01 --epochs 100 --dropout 0

## Dataset (data/music)
* `data.train, data.eval, data.test`
  * Training, Validation, Testing rating file.
  * Each line is a triple: ('User ID'  'Item ID'  'Rating').

* `data.para`
  * statistics of data.

* `data.user_graph, data.item_graph`
  * Review graph file of the user, item.

## Citation
```
@article{liu2021learning,
  title={Learning Hierarchical Review Graph Representations for Recommendation},
  author={Liu, Yong and Yang, Susen and Zhang, Yinan and Miao, Chunyan and Nie, Zaiqing and Zhang, Juyong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
```
