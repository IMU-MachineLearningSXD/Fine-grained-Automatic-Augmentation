import argparse
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets
from lib.utils.utils import model_info
from tensorboardX import SummaryWriter
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from lib.dataset import data_train_aug_load
from torchvision import transforms
import copy


# 载入config文件 有自定义的cfg命令
def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    # 这里是cfg的命令
    parser.add_argument('--cfg', default='./lib/config/OWN_Test_config.yaml', help='experiment configuration filename', required=False, type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config


# 定义objective
def objective(trial):

    # 载入 config文件
    config = parse_arg()

    # 输出文件  这里到时候应该可以删掉，先不要输出文件了
    output_dict = utils.create_log_folder(config, phase='train')

    # cuda设置 加快计算速度
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    model = crnn.get_crnn(config)

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    # 定义模型，放到cuda上
    model = model.to(device)

    # 定义CTCloss
    criterion = torch.nn.CTCLoss()

    # 初始的epoch，如果不断点续训就是0
    last_epoch = config.TRAIN.BEGIN_EPOCH

    # 测试optuna调参，仅调整一个lr 如果不调lr就把1解除注释，然后把2、3、4、5都注释掉
    # 1
    optimizer = utils.get_optimizer(config, model)
    # # 2
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # 自动调参的参数：学习率
    # # 3
    # optimizer = None
    # # 4
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # # 5
    # # 优化器的初始化
    # for group in optimizer.param_groups:
    #     group.setdefault('initial_lr', group['lr'])

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    # 输出模型的网络结构以及初始化参数，可以删除掉，不输出了
    # model_info(model)

    # 定义optuna策略库
    list_policy = []
    num_aug = 20   # 增广次数
    for i in range(num_aug):
        list_policy.append(trial.suggest_float('policy_%d' % (i+1), 0.1, 0.7, log=True))   # 左极限不能为0，否则报错

    # 源代码!!!!!!!!被注释掉的
    # train_dataset = get_dataset(config)(config, is_train=True)

    # 改train_dataset
    original_train_dataset_list, orig_labels = data_train_aug_load.get_train_data_list(config)
    aug_data, orig_labels = data_train_aug_load.bezier_aug(original_train_dataset_list, orig_labels, list_policy)
    train_dataset = data_train_aug_load.get_train_data(config)(aug_data, orig_labels)

    # 就把train_dataset 改好了就行，其它测试集和train_loader都不用改
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
        )
    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
        )

    best_acc = 0
    acc = 0.

    # 利用字典将字符转标签的方法
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        # 训练返回一个loss
        # 这里改一下train_dataset
        train_loss = function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device,
                                    epoch, writer_dict, output_dict)
        lr_scheduler.step()

        # 到预定的epoch就开始测试
        if epoch % config.TEST.FREQUENCY == 0:
            # 用到function
            acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch,
                                    writer_dict, output_dict)

            if acc > best_acc:
                best_acc = acc
                print("best acc is:", best_acc)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        # "optimizer": optimizer.state_dict(),
                        # "lr_scheduler": lr_scheduler.state_dict(),
                        "best_acc": best_acc,
                    }, os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
                )
        # save checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    # "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                    "best_acc": best_acc,
                }, os.path.join(output_dict['chs_dir'], "checkpoint_{}_loss_{:.4f}.pth".format(epoch, train_loss))
            )

        # trial.report(acc, epoch)  # 实验精度的汇报

        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    writer_dict['writer'].close()

    return best_acc


if __name__ == '__main__':

    n_trials = 100
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())   # 创建实验：要求最大化返回的结果
    study.optimize(objective, n_trials=n_trials)    # 跑满n_trials次实验会停止!!!!!!!! 这个要调整

    # 以下只是为了输出数据
    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))  # 总共完成的实验数量
    # print("  Number of pruned trials: ", len(pruned_trials))  # 忽略不跑的实验数量
    # print("  Number of complete trials: ", len(complete_trials))  # 实际跑的实验数量

    # 输出一下排序的值
    kk = []
    len_result = len(complete_trials)
    for i in range(len_result):
        kk.append([])
    for i in range(len_result):
        kk[i].append(complete_trials[i].value)
        kk[i].append(i + 1)
        kk[i].append(complete_trials[i].params)

    # 根据值进行排序
    paixu = sorted(kk, key=(lambda x: x[0]), reverse=True)
    # 输出
    print('输出所有结果')
    for i in range(0, len(paixu)):
        print(paixu[i])

    # 输出最好的值
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():  # 最后一段的输出
        print("    {}: {}".format(key, value))