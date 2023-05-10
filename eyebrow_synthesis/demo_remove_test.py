import os, sys

#print('응애')

os.chdir('eyebrow_synthesis') # 경로 이동
terminal_command = 'python test.py --dataroot ./datasets/test_data --name eyebrow_remove --model pix2pix --direction AtoB'
os.system(terminal_command)

os.chdir('..') # 다시 빠져나오기 