# python train.py --cuda --method opSgd_lopAdam --model logreg --epochs 20 --dir lopAdam --save --beta 0.001
# python train.py --cuda --method opAdam_lopAdam --model logreg --epochs 20 --dir lopAdam --save --beta 0.0000001

#python train.py --cuda --method opSgd_lopAdam --model vgg --epochs 200 --dir lopAdam --save --beta 0.001
# python train.py --cuda --method opAdam_lopAdam --model vgg --epochs 200 --dir lopAdam --save --beta 1e-8
for optim in []:
do
	for alpha0 in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
    do
    	python train.py --cuda --method $optim --model vgg --dir Alpha_exp --save --beta 0.001 --alpha $alpha0 lossthreshold
    done
done


