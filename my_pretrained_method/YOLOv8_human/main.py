import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import numpy
import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def train(args, params):
    # Model
    model = nn.yolo_v8_n(len(params['names']))
    model = util.load_weight('./weights/v8_n.pt', model)
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)

    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    for filename in os.listdir('../Dataset/CrowdHuman/images/train'):
        filenames.append('../Dataset/CrowdHuman/images/train/' + filename)

    sampler = None
    dataset = Dataset(filenames, args.input_size, params, True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=4, pin_memory=True, collate_fn=Dataset.collate_fn)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Start training
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    with open('weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch',
                                                   'box', 'dfl', 'cls',
                                                   'Recall', 'Precision', 'mAP@50', 'mAP'])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
            if args.local_rank == 0:
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_dfl_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            # Scheduler
            scheduler.step()

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)
                writer.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'mAP': str(f'{last[0]:.3f}'),
                                 'mAP@50': str(f'{last[1]:.3f}'),
                                 'Recall': str(f'{last[2]:.3f}'),
                                 'Precision': str(f'{last[2]:.3f}')})
                f.flush()

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                save = {'model': copy.deepcopy(ema.ema).half()}

                # Save last, best and delete
                torch.save(save, './weights/last.pt')
                if best == last[0]:
                    torch.save(save, './weights/best.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    for filename in os.listdir('../Dataset/CrowdHuman/images/val'):
        filenames.append('../Dataset/CrowdHuman/images/val/' + filename)
    numpy.random.shuffle(filenames)
    dataset = Dataset(filenames, args.input_size, params, augment=False)
    loader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs, 0.001, 0.7)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat((cls, util.wh2xy(box) * scale), 1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
    # Print results
    print(('%10s' + '%10.3g' * 4) % ("", m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


@torch.no_grad()
def demo(args):
    import cv2

    # Load model
    model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()
    model.half()
    model.eval()

    camera = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while camera.isOpened():
        # Capture frame-by-frame
        success, frame = camera.read()
        if success:
            image = frame.copy()
            shape = image.shape[:2]

            r = args.input_size / max(shape[0], shape[1])
            if r != 1:
                resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
            height, width = image.shape[:2]

            # Scale ratio (new / old)
            r = min(1.0, args.input_size / height, args.input_size / width)

            # Compute padding
            pad = int(round(width * r)), int(round(height * r))
            w = numpy.mod((args.input_size - pad[0]), 32) / 2
            h = numpy.mod((args.input_size - pad[1]), 32) / 2

            if (width, height) != pad:  # resize
                image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border

            # Convert HWC to CHW, BGR to RGB
            x = image.transpose((2, 0, 1))[::-1]
            x = numpy.ascontiguousarray(x)
            x = torch.from_numpy(x)
            x = x.unsqueeze(dim=0)
            x = x.cuda()
            x = x.half()
            x = x / 255
            # Inference
            outputs = model(x)
            # NMS
            outputs = util.non_max_suppression(outputs, 0.25, 0.7)
            for output in outputs:
                output[:, [0, 2]] -= w  # x padding
                output[:, [1, 3]] -= h  # y padding
                output[:, :4] /= min(height / shape[0], width / shape[1])

                output[:, 0].clamp_(0, shape[1])  # x1
                output[:, 1].clamp_(0, shape[0])  # y1
                output[:, 2].clamp_(0, shape[1])  # x2
                output[:, 3].clamp_(0, shape[0])  # y2

                for box in output:
                    box = box.cpu().numpy()
                    x1, y1, x2, y2, score, index = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    camera.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def profile(args, params):
    model = nn.yolo_v8_n(len(params['names']))
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    model(torch.zeros(shape))
    params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(f'Number of parameters: {int(params)}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.demo:
        demo(args)


if __name__ == "__main__":
    main()
