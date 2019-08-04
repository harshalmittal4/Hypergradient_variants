<<<<<<< HEAD
# new exp
# python train.py --cuda --method opSgd_lopAdam --model logreg --epochs 20 --dir lopAdam --save --beta 0.001
# python train.py --cuda --method opAdam_lopAdam --model logreg --epochs 20 --dir lopAdam --save --beta 0.0000001
=======
python train.py --cuda --method opSgd_lopAdam --model logreg --epochs 20 --dir lopAdam --save --beta 0.001
python train.py --cuda --method opAdam_lopAdam --model logreg --epochs 20 --dir lopAdam --save --beta 0.0000001
>>>>>>> parent of 4eb906b... Add prev exp left to new.sh

# python train.py --cuda --method opSgd_lopAdam --model vgg --epochs 200 --dir lopAdam --save --beta 0.001
python train.py --cuda --method opAdam_lopAdam --model vgg --epochs 200 --dir lopAdam --save --beta 1e-8



