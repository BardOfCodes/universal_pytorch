#############################################################
import utils
import torch.backends.cudnn as cudnn
cudnn.enabled = False
##############################################################
from docopt import docopt
import time
import torch
import torchvision
import numpy as np
import torch.optim as optim

docstr = """Find Universal Adverserial Perturbation for Image Classification models trained in pytorch.

Usage:
  find_uap.py <model> <im_path> <im_list> [options]
  find_uap.py (-h | --help)
  find_uap.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --data_dep=<bool>           Use data for finding UAP or not.[default: False]
  --save_loc=<str>            Location for saving the UAP as FloatTensor[default: same_dir]
  --batch_size=<int>          batch_size for processing while forming UAP in gpu[default: 25]
  --gpu=<bool>                Which GPU to use[default: 3]
  --max_iter_uni=<int>        maximum epochs to train for[default: 10]   
  --xi=<float>                controls the l_p magnitude of the perturbation[default: 0.1866]
  --delta=<float>             controls the desired fooling rate[default: 0.2]
  --p=<float>                 norm to be used for the UAP[default: inf]
  --num_classes=<int>         For deepfool: num_classes (limits the number of classes to test against)[default: 10]
  --overshoot=<float>         For deepfool: used as a termination criterion to prevent vanishing updates[default: 0.02]
  --max_iter_df=<int>         For deepfool: maximum number of iterations for deepfool[default: 10]
  --t_p=<float>               For batch deepfool: truth perentage, for how many flipped labels in a batch atleast.[default: 0.2]
"""

if __name__ == '__main__':
    start_time = time.time() 
    args = docopt(docstr, version='v1.0')
    torch.cuda.set_device(int(args['--gpu']))
    
    net = utils.get_model(args['<model>'])

    location_img = args['<im_path>']
    img_list = args['<im_list>']
    max_iter_uni=int(args['--max_iter_uni'])   
    xi=float(args['--xi'])
    delta=float(args['--delta'])
    if(args['--p'] == 'inf'):
        p = np.inf
    else:
        p=int(args['--p'])
    if(args['--save_loc'] == 'same_dir'):
        save_loc = '.'
    else:
        save_loc = args['--save_loc'] 
    num_classes=int(args['--num_classes'])
    overshoot=float(args['--overshoot'])
    max_iter_df=int(args['--max_iter_df'])
    t_p=float(args['--t_p'])
    
    file = open(img_list)
    img_names = []
    for f in file:
        img_names.append(f.split(' ')[0])
    img_names = [location_img +x for x in img_names]
    st = time.time()
    if(eval(args['--data_dep'])):
        batch_size = 1
        uap = utils.universal_perturbation_data_dependant(img_names, net, xi=xi, delta=delta, max_iter_uni =max_iter_uni,
                                                          p=p, num_classes=num_classes, overshoot=overshoot, 
                                                          max_iter_df=max_iter_df,init_batch_size = batch_size,t_p = t_p)
    else:
        batch_size = int(args['--batch_size'])
        uap = utils.universal_perturbation_data_independant(img_names, net,delta=delta, max_iter_uni = max_iter_uni, xi=xi,
                                                            p=p, num_classes=num_classes, overshoot=overshoot,
                                                            max_iter_df=max_iter_df,init_batch_size=batch_size)
        
    print('found uap.Total time: ' ,time.time()-st)
    uap = uap.data.cpu()
    torch.save(uap,save_loc+'perturbation_'+args['<model>']+'.pth')
