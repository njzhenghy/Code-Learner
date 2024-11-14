@echo off
call venv\Scripts\activate.bat
pythonw train_vit.py --dataset cifar100 --batch_size 64 --weight_decay 1e-5 --img_size 32 --gpu 0 --seed 22 
exit