# new exp
python train.py --cuda --method opSgd_lopAdam --model logreg --epochs 20 --dir lopAdam --save --beta 0.001
python train.py --cuda --method opAdam_lopAdam --model logreg --epochs 20 --dir lopAdam --save --beta 0.0000001

python train.py --cuda --method opSgd_lopAdam --model vgg --epochs 200 --dir lopAdam --save --beta 0.001
python train.py --cuda --method opAdam_lopAdam --model vgg --epochs 200 --dir lopAdam --save --beta 1e-8

#prev wale
python train.py --cuda --method opSgdn_lopSgdn --mu 0.9 --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method opSgdn_lopAdam --mu 0.9 --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method sgd --model logreg --epochs 20 --dir test --save







