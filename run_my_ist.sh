# ### 2 Workers Case
# python my_distributed_3layer.py --mask-ratio 0.7 --world-size 2 --rank 0 & python my_distributed_3layer.py --mask-ratio 0.7 --world-size 2 --rank 1
### 2 Workers Case
python my_distributed_3layer.py --world-size 2 --rank 0 & python my_distributed_3layer.py --world-size 2 --rank 1

# ### 3 Workers Case
# python my_distributed_3layer.py --world-size 3 --rank 0 & python my_distributed_3layer.py --world-size 3 --rank 1 & python my_distributed_3layer.py --world-size 3 --rank 2 

# ### 4 Workers Case
# python my_distributed_3layer.py --world-size 4 --rank 0 & python my_distributed_3layer.py --world-size 4 --rank 1 & python my_distributed_3layer.py --world-size 4 --rank 2 & python my_distributed_3layer.py --world-size 4 --rank 3