import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths

parser = argparse.ArgumentParser()
parser.add_argument('-p1', '--path1', default='sample/samples')
parser.add_argument('-p2', '--path2', default='/root/data/celeba-128/celeba-128')
args = parser.parse_args()
fid=calculate_fid_given_paths((f'{args.path1}',f'{args.path2}'),128,'cuda',2048,8)
print(fid)