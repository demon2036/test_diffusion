from pytorch_fid.fid_score import calculate_fid_given_paths

fid=calculate_fid_given_paths(('./sample/data',''),128,'cuda',2048,8)
print(fid)