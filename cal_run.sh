python train.py --cuda --method sgdn --mu 0.9 --model vgg --epochs 200 --dir test --save
python train.py --cuda --method opSgdn_lopHd --mu 0.9 --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method opSgdn_lopSgdn --mu 0.9 --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method opSgdn_lopAdam --mu 0.9 --model vgg --epochs 200 --dir test --save --beta 0.001

