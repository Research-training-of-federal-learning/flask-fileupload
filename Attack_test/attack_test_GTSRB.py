import argparse
import shutil
from datetime import datetime

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from Attack.dataset.pipa import Annotations  # legacy to correctly load dataset.
from Attack.helper import Helper
from Attack.utils.utils import *

logger = logging.getLogger('logger')


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True):
    criterion = hlpr.task.criterion
    model.train() # 模型切换到训练模式  model 指的是建立好的GTSRB模型网络


    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)# 在train_loader中迭代每一个batch，使用hlpr.task.get_batch(i, data)将batch处理为模型可以处理的格式，并存储在batch变量中
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        loss.backward()#计算梯度
        optimizer.step()#更新模型参数

        hlpr.report_training_losses_scales(i, epoch)#记录训练过程中的损失和scale
        if i == hlpr.params.max_batch_id:
            break

    return


def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model # 获取要测试的模型
    model.eval() # 将模型设置为评估模式，即在测试时不进行梯度计算
    hlpr.task.reset_metrics() # 将存储度量的列表清空


    with torch.no_grad(): # 关闭梯度计算上下文，这样可以在前向传递过程中节省内存，并且不会为后面的反向传递过程分配内存。
        for i, data in tqdm(enumerate(hlpr.task.test_loader)): #对测试集进行迭代，同时记录迭代次数i和对应的数据data。
            #print(1)
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels) # 将当前预测输出和对应的标签存储到度量的列表中。
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric


def run(hlpr):
    acc = test(hlpr, 0, backdoor=False)
    acc_b = test(hlpr, 0, backdoor=True)
    return acc,acc_b

def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric)


def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)

    hlpr.task.update_global_model(weight_accumulator, global_model)

def main(paramspath,name):

    with open(paramspath) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    # params['commit'] = args.commit
    params['name'] = name

    helper = Helper(params)
    logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)
        else:
            acc,acc_b = run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'Attack/runs/{name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {name}")
        else:
            logger.error(f"Aborted training. No output generated.")
    return acc,acc_b

if __name__ == '__main__':
    print("start_GTSRB!!!")
    main("configs/gtsrb_params.yaml", "gtsrb")
    # parser = argparse.ArgumentParser(description='Backdoors')
    # parser.add_argument('--params', dest='params', default='utils/params.yaml')
    # parser.add_argument('--name', dest='name', required=True) # 必选参数
    # parser.add_argument('--commit', dest='commit',
    #                     default=get_current_git_hash())
    #
    # args = parser.parse_args() # 解析命令行参数
    #
    # with open(args.params) as f:
    #     params = yaml.load(f, Loader=yaml.FullLoader) # 使用yaml模块读取params文件中的数据，并将其存储在params变量中。
    #
    # params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    # params['commit'] = args.commit
    # params['name'] = args.name
    #
    # helper = Helper(params)
    # logger.warning(create_table(params)) # 调用create_table()函数生成一张数据表，并使用logger模块的warning()函数将表格写入日志文件中。
    #
    # try:
    #     if helper.params.fl:
    #         fl_run(helper)
    #     else:
    #         run(helper)
    # except (KeyboardInterrupt):
    #     if helper.params.log:
    #         answer = prompt('\nDelete the repo? (y/n): ')
    #         if answer in ['Y', 'y', 'yes']:
    #             logger.error(f"Fine. Deleted: {helper.params.folder_path}")
    #             shutil.rmtree(helper.params.folder_path)
    #             if helper.params.tb:
    #                 shutil.rmtree(f'runs/{args.name}')
    #         else:
    #             logger.error(f"Aborted training. "
    #                          f"Results: {helper.params.folder_path}. "
    #                          f"TB graph: {args.name}")
    #     else:
    #         logger.error(f"Aborted training. No output generated.")
