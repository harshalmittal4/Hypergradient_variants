python train.py --cuda --method opSgd_lopHd --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method opSgd_lopSgdn --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method opSgd_lopAdam --model vgg --epochs 200 --dir test --save --beta 0.001

