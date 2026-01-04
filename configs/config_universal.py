from datetime import datetime
from pathlib import Path
from project_utils import *

class UniversalConfig:
    def __init__(self, execute_model_index, execute_pretrain_model_index, execute_dataset_index, criterion=None, optimizer=None, scheduler=None, num_workers=None):
        self.support_model_list = ['SS_Net']
        self.execute_model = self.support_model_list[execute_model_index]
        self.project_directory = Path.cwd()
        self.execute_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.support_pretrain_list = [None, 'min_loss.pth', 'max_score.pth', 'max_dice.pth', 'latest.pth']
        self.execute_pretrain_model = self.support_pretrain_list[execute_pretrain_model_index]
        if self.execute_pretrain_model is not None:
            self.pretrain_model_path = self.project_directory / 'pretrain' / self.execute_model / self.execute_pretrain_model
        else:
            self.pretrain_model_path = None

        #                            0         1         2         3         4         5           6            7          8           9           10         11         12         13         14          15
        self.support_dataset_list = ['BUSI_1', 'BUSI_2', 'BUSI_3', 'BUSI_4', 'BUSI_5', 'BrEaST_1', 'BrEaST_2', 'BrEaST_3', 'BrEaST_4', 'BrEaST_5', 'UDIAT_1', 'UDIAT_2', 'UDIAT_3', 'UDIAT_4', 'UDIAT_5', 'STU']
        self.execute_dataset = self.support_dataset_list[execute_dataset_index]
        self.dataset_directory = self.project_directory / 'datasets' / self.execute_dataset

        self.result_directory_name = self.execute_model + '__' + self.execute_dataset + '__' + self.execute_time
        self.result_directory = self.project_directory / 'results' / self.result_directory_name
        self.log_directory = self.result_directory / 'log'
        self.output_directory = self.result_directory / 'output'
        self.best_models_directory = self.result_directory / 'best_models'


        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda':
            self.automatic_mixed_precision = False
        else:
            self.automatic_mixed_precision = False

        self.num_classes = 1
        self.input_size_h = 256
        self.input_size_w = 256
        self.input_channels = 3
        self.seed = 1202
        self.batch_size = 16
        self.total_epochs = 1000
        self.evaluate_threshold = 0.5
        self.gradient_clipping = 1
        self.early_stopping_patience = 125
        if num_workers is None:
            self.num_workers = 16
        else:
            self.num_workers = num_workers

        self.print_interval = 5
        self.val_interval = 1
        self.result_interval = 0.05
        self.save_interval = 1
        self.estimate_interval = 2 * self.val_interval

        if criterion == 'BceDiceLoss':
            self.criterion = BceDiceLoss()
        elif criterion is None:
            self.criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5, n_labels=self.num_classes)

        if optimizer is None:
            self.optimizer = 'AdamW'
        else:
            self.optimizer = optimizer
        assert self.optimizer in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                                  'SGD'], 'Unsupported optimizer!'
        if self.optimizer == 'Adadelta':
            self.lr = 0.01  # default: 1.0 – coefficient that scale delta before it is applied to the parameters
            self.rho = 0.9  # default: 0.9 – 用于计算梯度平方的运行平均值的系数
            self.eps = 1e-6  # default: 1e-6 – term added to the denominator to improve numerical stability
            self.weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
        elif self.optimizer == 'Adagrad':
            self.lr = 0.01  # default: 0.01 – learning rate
            self.lr_decay = 0  # default: 0 – learning rate decay
            self.eps = 1e-10  # default: 1e-10 – term added to the denominator to improve numerical stability
            self.weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
        elif self.optimizer == 'Adam':
            self.lr = 0.001  # default: 1e-3 – learning rate
            self.betas = (0.9,
                          0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
            self.eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
            self.weight_decay = 0.0001  # default: 0 – weight decay (L2 penalty)
            self.amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
        elif self.optimizer == 'AdamW':
            self.lr = 1e-4  # default: 1e-3 – learning rate
            self.betas = (0.9,
                          0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
            self.eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
            self.weight_decay = 1e-2  # default: 1e-2 – weight decay coefficient
            self.amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
        elif self.optimizer == 'Adamax':
            self.lr = 2e-3  # default: 2e-3 – learning rate
            self.betas = (0.9,
                          0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
            self.eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
            self.weight_decay = 0  # default: 0 – weight decay (L2 penalty)
        elif self.optimizer == 'ASGD':
            self.lr = 0.01  # default: 1e-2 – learning rate
            self.lambd = 1e-4  # default: 1e-4 – decay term
            self.alpha = 0.75  # default: 0.75 – power for eta update
            self.t0 = 1e6  # default: 1e6 – point at which to start averaging
            self.weight_decay = 0  # default: 0 – weight decay
        elif self.optimizer == 'RMSprop':
            self.lr = 1e-2  # default: 1e-2 – learning rate
            self.momentum = 0  # default: 0 – momentum factor
            self.alpha = 0.99  # default: 0.99 – smoothing constant
            self.eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
            self.centered = False  # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
            self.weight_decay = 0  # default: 0 – weight decay (L2 penalty)
        elif self.optimizer == 'Rprop':
            self.lr = 1e-2  # default: 1e-2 – learning rate
            self.etas = (0.5,
                         1.2)  # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
            self.step_sizes = (1e-6, 50)  # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes
        elif self.optimizer == 'SGD':
            self.lr = 1e-6  # – learning rate
            self.momentum = 0.9  # default: 0 – momentum factor
            self.weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
            self.dampening = 0  # default: 0 – dampening for momentum
            self.nesterov = False  # default: False – enables Nesterov momentum

        if optimizer is None:
            self.scheduler = 'CosineAnnealingLR'
        else:
            self.scheduler = scheduler
        assert self.scheduler in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                                  'CosineAnnealingWarmRestarts',
                                  'WP_MultiStepLR', 'WP_CosineLR', 'AdaptiveLinearAnnealingSoftRestarts'], 'Unsupported scheduler! '
        if self.scheduler == 'StepLR':
            self.step_size = self.total_epochs // 5  # – Period of learning rate decay.
            self.gamma = 0.5  # – Multiplicative factor of learning rate decay. Default: 0.1
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'MultiStepLR':
            self.milestones = [60, 120, 150]  # – List of epoch indices. Must be increasing.
            self.gamma = 0.1  # – Multiplicative factor of learning rate decay. Default: 0.1.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'ExponentialLR':
            self.gamma = 0.99  # – Multiplicative factor of learning rate decay.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'CosineAnnealingLR':
            self.T_max = 50  # – Maximum number of iterations. Cosine function period.
            self.eta_min = 0.00001  # – Minimum learning rate. Default: 0.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'ReduceLROnPlateau':
            self.mode = 'min'  # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
            self.factor = 0.1  # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
            self.patience = 10  # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
            self.threshold = 0.0001  # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
            self.threshold_mode = 'rel'  # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
            self.cooldown = 0  # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
            self.min_lr = 0  # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
            self.eps = 1e-08  # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
        elif self.scheduler == 'CosineAnnealingWarmRestarts':
            self.T_0 = 10  # – Number of iterations for the first restart.
            self.T_mult = 1  # – A factor increases T_{i} after a restart. Default: 1.
            self.eta_min = 1e-4  # – Minimum learning rate. Default: 0.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'WP_MultiStepLR':
            self.warm_up_epochs = 10
            self.gamma = 0.1
            self.milestones = [125, 225]
        elif self.scheduler == 'WP_CosineLR':
            self.warm_up_epochs = 20

