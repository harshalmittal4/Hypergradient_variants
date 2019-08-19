
# for insensitivity to alpha_0 experiments
for optim in 'sgd_Hd' 'op_sgd_lop_sgdn' 'sgd' 'sgdn_Hd' 'op_sgdn_lop_sgdn' 'sgdn'
do
	
	if  [ $optim = 'sgd_Hd' ] || [ $optim = 'op_sgd_lop_sgdn' ] || [ $optim = 'sgd' ]
	then
		t=0.75
	else
		t=0.3
	fi
	
	for alpha0 in 0.01 0.005 0.001 0.0005 0.0001 
    do
    	python train.py --cuda --method $optim --model vgg --dir Alpha_exp --save --beta 0.001 --alpha_0 $alpha0 --lossThreshold $t
    done
done


for optim in 'op_adam_lop_adam' 'adam' 'adam_Hd'
do
	for alpha0 in 0.01 0.005 0.001 0.0005 0.0001
    do
    	python train.py --cuda --method $optim --model vgg --dir Alpha_exp --save --beta 1e-8 --alpha $alpha0 --lossThreshold 0.2
    done
done