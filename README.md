# HD-varians

To run:
1) Create and activate venv (python3)
2) pip install -r requirements.txt
3) python train.py --cuda --method opAdam_lopSgd --model logreg --epochs 50 --dir test --save --beta 0.001

4) Hyperparams:
{<br>
'logreg':{'alpha0':0.001, 'beta': 0.001 for SGD-HD and SGDN-HD, and β = 10−7 for Adam-HD, 'epochs': 20},<br>
'mlp':{'alpha0':0.001, 'beta': 0.001 for SGD-HD and SGDN-HD, and β = 10−7 for Adam-HD., 'epochs': 100},<br>
'vgg':{'alpha0':0.001, 'beta': 0.001 for SGD-HD and SGDN-HD, and β = 10−8 for Adam-HD, 'epochs': 200}}<br>
