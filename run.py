import sys
from dla import dla
if __name__ == '__main__':
    
    args = sys.argv[1].split(',')
    r = int(args[0])
    N = int(args[1])
    k = float(args[2])
    
    
    obj = dla(demo=False,radius_n_k= [r,N,k])
    b = obj.run_dla(early_stop=True,save_steps=True)
    print('ended : ', b)
    