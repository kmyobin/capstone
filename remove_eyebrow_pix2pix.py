import subprocess

command = "python ./train_eyebrow/test.py --dataroot ./train_eyebrow/datasets/eyebrows --name tmp_pix2pix --model pix2pix --direction AtoB"
subprocess.run(command.split())