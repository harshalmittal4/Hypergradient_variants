python train.py --cuda --method opSgd_lopHd --model logreg --epochs 20 --dir test --save --beta 0.001
python train.py --cuda --method opSgd_lopSgdn --model logreg --epochs 20 --dir test --save --beta 0.001
python train.py --cuda --method opSgd_lopAdam --model logreg --epochs 20 --dir test --save --beta 0.001

python train.py --cuda --method sgdn --mu 0.9 --model logreg --epochs 20 --dir test --save
python train.py --cuda --method opSgdn_lopHd --mu 0.9 --model logreg --epochs 20 --dir test --save --beta 0.001
python train.py --cuda --method opSgdn_lopSgdn --mu 0.9 --model logreg --epochs 20 --dir test --save --beta 0.001
python train.py --cuda --method opSgdn_lopAdam --mu 0.9 --model logreg --epochs 20 --dir test --save --beta 0.001

python train.py --cuda --method adam --model logreg --epochs 20 --dir test --save
python train.py --cuda --method opAdam_lopHd --model logreg --epochs 20 --dir test --save --beta 0.0000001
python train.py --cuda --method opAdam_lopSgdn --model logreg --epochs 20 --dir test --save --beta 0.0000001
python train.py --cuda --method opAdam_lopAdam --model logreg --epochs 20 --dir test --save --beta 0.0000001
