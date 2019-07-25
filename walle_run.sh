python train.py --cuda --method adam --model vgg --epochs 200 --dir test --save
python train.py --cuda --method opAdam_lopHd --model vgg --epochs 200 --dir test --save --beta 1e-8
python train.py --cuda --method opAdam_lopSgdn --model vgg --epochs 200 --dir test --save --beta 1e-8
python train.py --cuda --method opAdam_lopAdam --model vgg --epochs 200 --dir test --save --beta 1e-8


