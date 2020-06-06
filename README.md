# Medical Robust Training

## Basic usage

* Network standard training
```
python main.py --affix dsbn_res50_std --batch_size 16    
```
 
* Network robust training
```
python main.py --adv_train --affix dsbn_res50_linf --batch_size 16 --max_epoch 250    
```

* Network testing and plot ROC-AUC
```
python main.py --affix dsbn_res18_AUC --todo test --load_checkpoint ./checkpoint/rijeka_ROIMG_dsbn_res18/checkpoint_best.pth 
```
