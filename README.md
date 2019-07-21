# HD-varians

To run:
1) Create and activate venv (python3)
2) pip install -r requirements.txt<br>
3) 
SGD<br>
python train.py --cuda --method sgd --model logreg --epochs 20 --dir test --save<br>
python train.py --cuda --method opSgd_lopHd --model logreg --epochs 20 --dir test --save --beta 0.001<br>
python train.py --cuda --method opSgd_lopSgdn --model logreg --epochs 20 --dir test --save --beta 0.001<br>
python train.py --cuda --method opSgd_lopAdam --model logreg --epochs 20 --dir test --save --beta 0.001<br>
<br>
SGDN<br>
python train.py --cuda --method sgdn --mu 0.9 --model logreg --epochs 20 --dir test --save<br>
python train.py --cuda --method opSgdn_lopHd --mu 0.9 --model logreg --epochs 20 --dir test --save --beta 0.001<br>
python train.py --cuda --method opSgdn_lopSgdn --mu 0.9 --model logreg --epochs 20 --dir test --save --beta 0.001<br>
python train.py --cuda --method opSgdn_lopAdam --mu 0.9 --model logreg --epochs 20 --dir test --save --beta 0.001<br>
<br>
Adam<br>
python train.py --cuda --method adam --model logreg --epochs 20 --dir test --save<br>
python train.py --cuda --method opAdam_lopHd --model logreg --epochs 20 --dir test --save --beta 0.0000001<br>
python train.py --cuda --method opAdam_lopSgdn --model logreg --epochs 20 --dir test --save --beta 0.0000001<br>
python train.py --cuda --method opAdam_lopAdam --model logreg --epochs 20 --dir test --save --beta 0.0000001<br>

--model vgg/mlp --epochs 200/100 (rest same)


4) Hyperparams:
{<br>
'logreg':{'alpha0':0.001, 'beta': 0.001 for SGD-HD and SGDN-HD, and β = 10−7 for Adam-HD, 'epochs': 20},<br>
'mlp':{'alpha0':0.001, 'beta': 0.001 for SGD-HD and SGDN-HD, and β = 10−7 for Adam-HD., 'epochs': 100},<br>
'vgg':{'alpha0':0.001, 'beta': 0.001 for SGD-HD and SGDN-HD, and β = 10−8 for Adam-HD, 'epochs': 200}}<br>


