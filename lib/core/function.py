from  __future__ import  absolute_import
import time
import lib.utils.utils as utils
import torch
from tqdm import tqdm


# 算个平均数
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 返回平均loss
def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    with tqdm(total=len(train_loader), desc='epoch {}'.format(epoch), unit='steps') as bar:
        for i, (inp, idx) in enumerate(train_loader):

            # measure data time
            data_time.update(time.time() - end)

            labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)

            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)                    # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标

            preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
            loss = criterion(preds, text, preds_size, length)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), inp.size(0))

            batch_time.update(time.time()-end)

            bar.set_postfix_str('loss: {}'.format(losses.avg))
            # if i % config.PRINT_FREQ == 0:
            #     msg = 'Epoch: [{0}][{1}/{2}]\t' \
            #           'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #           'Speed {speed:.1f} samples/s\t' \
            #           'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #           'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
            #               epoch, i, len(train_loader), batch_time=batch_time,
            #               speed=inp.size(0)/batch_time.val,
            #               data_time=data_time, loss=losses)
            #     print(msg)
            #
            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            end = time.time()
            bar.update(1)

    return losses.avg


# 返回的是一个精度
def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict):

    losses = AverageMeter()
    model.eval()

    n_correct = 0
    with torch.no_grad():
        # 这里就会实时输出测试进度
        with tqdm(total=len(val_loader), desc='val epoch {}'.format(epoch), unit='steps') as bar:
            for i, (inp, idx) in enumerate(val_loader):

                labels = utils.get_batch_label(dataset, idx)
                inp = inp.to(device)

                # inference
                preds = model(inp).cpu()

                # compute loss
                batch_size = inp.size(0)
                text, length = converter.encode(labels)
                preds_size = torch.IntTensor([preds.size(0)] * batch_size)
                loss = criterion(preds, text, preds_size, length)

                losses.update(loss.item(), inp.size(0))
                bar.set_postfix_str('loss: {}'.format(losses.avg))

                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
                for pred, target in zip(sim_preds, labels):
                    if pred == target:
                        n_correct += 1

                # if (i + 1) % config.PRINT_FREQ == 0:
                #     print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

                if i == config.TEST.NUM_TEST_BATCH:
                    break

                bar.update(1)

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        # 输出测试label的表现，就是那一大串字符
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)

    # 输出（测试正确的/总共的） 这是一个batch下的
    print("[#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    accuracy = n_correct / float(num_test_sample)
    # 汇报loss和精度
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + config.TEST.FREQUENCY

    return accuracy