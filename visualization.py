import argparse
import logging
import os
import pprint
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from dataset.semi import SemiDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from model.model_helper import ModelBuilder

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--ckpt_path', type=str, required=True) # evaluate checkpoints


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = ModelBuilder(cfg['model'])

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    epoch = -1
    
    checkpoint = torch.load(os.path.join(args.ckpt_path, 'best.pth'))
    model.load_state_dict(checkpoint['model'])
    
    model.module.decoder.set_SMem_status(epoch=0, isVal=True)
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    
    print("start")
    
    iter_cnt = 0
    
    model.eval()
    with torch.no_grad():
        for get_input in valloader:
            img = get_input[0]
            mask = get_input[1]
            
            if iter_cnt < 2:
                iter_cnt = iter_cnt + 1
                continue
            
            #img, mask = img.cuda(), mask.cuda()
            #print(mask)
            #c1,o c2, c3, c4 = model(img, True)
            #print(model.decoder)
            pred = model(img, get_figure=True)
            pred = pred.squeeze()
            pred = pred.permute(1, 2, 0)
            #print(pred.shape)
            torch.save(pred,"figure_tensor.pt")
            pred = model(img)
            pred = pred.squeeze()
            print(pred)
            pred = pred.argmax(dim=0)
            torch.save(pred, "labeled.pt")
            quit()
            #print(c1.shape)
            #print(c2.shape)
            #print(c3.shape)
            #print(c4.shape)
            #print(pred.shape)
            #print(pred)
            '''
            c4 = torch.reshape(c4, [c4.shape[1],c4.shape[2],c4.shape[3]])
            c4 = c4.permute(1, 2, 0)
            H, W, D = c4.shape
            c4 = torch.reshape(c4, [H*W, D])
            c4 = c4.cpu().tolist()
            torch.save(c4,"figure_tensor.pt")
            break
            '''

if __name__ == '__main__':
    main()
