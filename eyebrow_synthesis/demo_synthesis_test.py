import os, sys

#print('응애')

os.chdir('eyebrow_synthesis') # 경로 이동
terminal_command = 'python test.py --dataroot ./datasets/test_data --name eyebrow_synthesis --model pix2pix --direction AtoB'
os.system(terminal_command)

