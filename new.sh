# python train.py --cuda --method opSgd_lopAdam --model mlp --epochs 100 --dir lopAdam --save --beta 0.001
# python train.py --cuda --method opAdam_lopAdam --model mlp --epochs 100 --dir lopAdam --save --beta 0.0000001

#python train.py --cuda --method opSgd_lopAdam --model vgg --epochs 1000 --dir lopAdam --save --beta 0.001
# python train.py --cuda --method opAdam_lopAdam --model vgg --epochs 1000 --dir lopAdam --save --beta 1e-8

#SGD
python train.py --cuda --method sgd --model mlp --epochs 100 --dir test --save
python train.py --cuda --method sgd_Hd --model mlp --epochs 100 --dir test --save --beta 0.001
python train.py --cuda --method sgd_Hdmomentum --model mlp --epochs 100 --dir test --save --beta 0.001
python train.py --cuda --method sgd_HdAdam --model mlp --epochs 100 --dir test --save --beta 0.001

#SGDN
python train.py --cuda --method sgdn --mu 0.9 --model mlp --epochs 100 --dir test --save
python train.py --cuda --method sgdn_Hd --mu 0.9 --model mlp --epochs 100 --dir test --save --beta 0.001
python train.py --cuda --method sgdn_Hdmomentum --mu 0.9 --model mlp --epochs 100 --dir test --save --beta 0.001
python train.py --cuda --method sgdn_HdAdam --mu 0.9 --model mlp --epochs 100 --dir test --save --beta 0.001

#Adam
python train.py --cuda --method adam --model mlp --epochs 100 --dir test --save
python train.py --cuda --method adam_Hd --model mlp --epochs 100 --dir test --save --beta 0.0000001
python train.py --cuda --method adam_Hdmomentum --model mlp --epochs 100 --dir test --save --beta 0.0000001
python train.py --cuda --method adam_HdAdam --model mlp --epochs 100 --dir test --save --beta 0.0000001


for optim in ['sgd_HdAdam', 'sgd_Hd', 'sgd_Hdmomentum', 'sgd', 'sgdn_Hd','sgdn_Hdmomentum', 'sgdn']:
do
	if optim in ['sgd_HdAdam', 'sgd_Hd', 'sgd_Hdmomentum', 'sgd']:
	then
		t = 0.75
	else
		t = 0.3
	fi
	for alpha0 in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
    do
    	python train.py --cuda --method $optim --model vgg --dir Alpha_exp --save --beta 0.001 --alpha_0 $alpha0 --lossThreshold $t
    done
done


for optim in ['adam_HdAdam', 'adam', 'adam_Hd']:
do
	for alpha0 in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
    do
    	python train.py --cuda --method $optim --model vgg --dir Alpha_exp --save --beta 1e-8 --alpha $alpha0 --lossThreshold 0.2
    done
done