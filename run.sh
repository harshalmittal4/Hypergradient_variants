
#logreg
python train.py --cuda --method sgd --model logreg --epochs 20 --dir test --save
python train.py --cuda --method sgd_Hd --model logreg --epochs 20 --dir test --save --beta 0.001
python train.py --cuda --method op_sgd_lop_sgdn --model logreg --epochs 20 --dir test --save --beta 0.001
python train.py --cuda --method sgdn --mu 0.9 --model logreg --epochs 20 --dir test --save
python train.py --cuda --method sgdn_Hd --mu 0.9 --model logreg --epochs 20 --dir test --save --beta 0.001
python train.py --cuda --method op_sgdn_lop_sgdn --mu 0.9 --model logreg --epochs 20 --dir test --save --beta 0.001
python train.py --cuda --method adam --model logreg --epochs 20 --dir test --save
python train.py --cuda --method adam_Hd --model logreg --epochs 20 --dir test --save --beta 0.0000001
python train.py --cuda --method op_adam_lop_adam --model logreg --epochs 20 --dir test --save --beta 0.0000001

#mlp
python train.py --cuda --method sgd --model mlp --epochs 100 --dir test --save
python train.py --cuda --method sgd_Hd --model mlp --epochs 100 --dir test --save --beta 0.001
python train.py --cuda --method op_sgd_lop_sgdn --model mlp --epochs 100 --dir test --save --beta 0.001
python train.py --cuda --method sgdn --mu 0.9 --model mlp --epochs 100 --dir test --save
python train.py --cuda --method sgdn_Hd --mu 0.9 --model mlp --epochs 100 --dir test --save --beta 0.001
python train.py --cuda --method op_sgdn_lop_sgdn --mu 0.9 --model mlp --epochs 100 --dir test --save --beta 0.001
python train.py --cuda --method adam --model mlp --epochs 100 --dir test --save
python train.py --cuda --method adam_Hd --model mlp --epochs 100 --dir test --save --beta 0.0000001
python train.py --cuda --method op_adam_lop_adam --model mlp --epochs 100 --dir test --save --beta 0.0000001

#vgg
python train.py --cuda --method sgd --model vgg --epochs 200 --dir test --save
python train.py --cuda --method sgd_Hd --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method op_sgd_lop_sgdn --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method sgdn --mu 0.9 --model vgg --epochs 200 --dir test --save
python train.py --cuda --method sgdn_Hd --mu 0.9 --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method op_sgdn_lop_sgdn --mu 0.9 --model vgg --epochs 200 --dir test --save --beta 0.001
python train.py --cuda --method adam --model vgg --epochs 200 --dir test --save
python train.py --cuda --method adam_Hd --model vgg --epochs 200 --dir test --save --beta 1e-8
python train.py --cuda --method op_adam_lop_adam --model vgg --epochs 200 --dir test --save --beta 1e-8