#############################################################
import utils
import torch.backends.cudnn as cudnn
cudnn.enabled = False
##############################################################
from docopt import docopt
import time
import torch
import torch.optim as optim

docstr = """Test Universal Adverserial Perturbation for Image Classification models trained in pytorch.

Usage:
  test_uap.py <model> <im_path> <im_list> <perturbation_path> [options]
  test_uap.py (-h | --help)
  test_uap.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --batch_size=<int>          batch_size for processing while forming UAP in gpu[default: 100]
  --gpu=<int>                Which GPU to use[default: 3]
"""

if __name__ == '__main__':
    start_time = time.time() 
    args = docopt(docstr, version='v1.0')
    torch.cuda.set_device(int(args['--gpu']))
    batch_size = int(args['--batch_size'])
    
    net = utils.get_model(args['<model>'])
    location_img = args['<im_path>']
    img_list = args['<im_list>']
    
    file = open(img_list)
    img_names = []
    for f in file:
        img_names.append(f.split(' ')[0])
    img_names = [location_img +x for x in img_names]
    st = time.time()
    v = torch.load(args['<perturbation_path>'])
    v = torch.autograd.Variable(v.cuda())
    _,_ = utils.get_fooling_rate(img_names,batch_size,v,net)
    
    print('Total time: ' ,time.time()-st)
