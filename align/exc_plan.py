import os
import time

from align.model_set import align_set
from args import configUtil
from args.configUtil import configClass

def rename_file(old_name, new_name):
    try:
        os.rename(old_name, new_name)
    except FileExistsError:
        print("文件名已存在")
    except Exception as e:
        # 访问异常的错误编号和详细信息
        print(e.args)

def runOneModel(model_name, dataset, hops=1, n_layers=1, peak_lr = 1e-4):
    # 使用 parse_args() 解析添加的参数
    args = configUtil.parse_args()
    args.dataset = dataset

    # 非代码执行流程
    args.device = "0"  # -1
    args.longterm_emb = 'path2/longterm_LaBSE.emb'
    args.peak_lr = peak_lr #1e-5 ->1e-4
    args.batch_size = 1024*1
    args.hops = hops
    args.n_layers = n_layers
    #args.pe_dim = 30
    args.patience = 20
    args.name = model_name

    # 模型执行开始
    print('---------------------------------------------')
    myconfig = configClass(args, os.path.realpath(__file__))
    ## 定义模型及运行
    mymodel = align_set(myconfig)
    outre, oldfilename, newfilename = mymodel.model_run()
    myconfig.myprint("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

    #
    rename_file(oldfilename, newfilename)
    print('---------------------------------------------')
    ##
    # outfile = '../../output202409/' + args.dataset +'/allResult_1009.txt'
    # with open(outfile, 'a', encoding='utf-8') as fw:
    #     fw.write('dataset:{}\n'.format(dataset))
    #     fw.write('model_name:{},hops:{}, layers:{}\n'.format(model_name, hops, n_layers))
    #     fw.write(outre + '\n\n')
    #     #fw.write('-'*20 + '\n')


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 这样可以保证报错信息提示的位置是准确的
    #dataset
    #datasets = 'DBP15K/fr_en/' # fr_en ja_en zh_en fr_en ja_en zh_en
    # datasets = 'WN31/EN_DE_15K_V1/'  # EN_DE_15K_V1、EN_FR_15K_V1
    # datasets = 'DWY100K/dbp_yg/' # DWY100K/dbp_wd, DWY100K/dbp_yg

    datasets_list = ['DBP15K/zh_en/', 'DBP15K/ja_en/','DBP15K/fr_en/']
    # datasets_list = ['WN31/EN_FR_15K_V1/', 'WN31/EN_FR_15K_V2/', 'WN31/EN_DE_15K_V1/', 'WN31/EN_DE_15K_V2/']
    for data in datasets_list:
        # runOneModel('0909_layers=3', data, n_layers=3)
        # peak_lr=1e-4, model_set2

        #layers1
        runOneModel('encoderLayer_hops1', data, hops=1, n_layers=1)
        runOneModel('encoderLayer_hops2', data, hops=2, n_layers=1)
        runOneModel('encoderLayer_hops3', data, hops=3, n_layers=1)
        runOneModel('encoderLayer_hops4', data, hops=4, n_layers=1)

        # layers2
        runOneModel('encoderLayer_hops1_layers2', data, hops=1, n_layers=2)
        runOneModel('encoderLayer_hops2_layers2', data, hops=2, n_layers=2)
        runOneModel('encoderLayer_hops3_layers2', data, hops=3, n_layers=2)
        runOneModel('encoderLayer_hops4_layers2', data, hops=4, n_layers=2)

        # layers3
        runOneModel('encoderLayer_hops1_layers3', data, hops=1, n_layers=3)
        runOneModel('encoderLayer_hops2_layers3', data, hops=2, n_layers=3)
        runOneModel('encoderLayer_hops3_layers3', data, hops=3, n_layers=3)
        runOneModel('encoderLayer_hops4_layers3', data, hops=4, n_layers=3)





