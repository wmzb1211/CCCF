import numpy as np
import torch
import yaml
from torch import optim, nn
import hdbscan
import argparse
from FederatedTask import Cifar10FederatedTask, TinyImagenetFederatedTask
from Params import Params
from Client import Client
from Server import ServerAvg

from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from torch.nn import Module
from Attacks import Attacks
# Type Definition
# define all the operations on global models
from synthesizers.pattern_synthesizer import PatternSynthesizer
from torch.utils.data import Subset
from Reports import FLReport, save_report, load_report


# random.seed(10)

class FederatedBackdoorExperiment:
    def __init__(self, params):
        # prepare for dataset and model
        self.params = params

        if params.task == 'CifarFed':
            self.task = Cifar10FederatedTask(params=params)
            print("Cifar!!!")
        elif params.task == 'ImageNetFed':
            self.task = TinyImagenetFederatedTask(params=params)
            print("ImageNet!!!")
        else:
            print("Not support dataset")
        print("Training Dataset:{}".format(params.task))
        self.task.init_federated_task()

        base_model = self.task.build_model()
        base_optimizer = self.task.build_optimizer(base_model)
        splited_dataset = self.task.sample_dirichlet_train_data(params.n_clients)
        print("splited_dataset:", len(splited_dataset))
        # print("splited_dataset:", splited_dataset)
        server_sample_ids = splited_dataset[params.n_clients - 1]
        print("server_sample_ids:", len(server_sample_ids))
        # print("server_sample_ids:", server_sample_ids)
        print("build server:", len(server_sample_ids))
        server_dataset = None
        if not len(server_sample_ids) == 0:
            server_dataset = Subset(self.task.train_dataset, server_sample_ids)
            print("server_dataset:", len(server_dataset))
            print("server_dataset:", server_dataset)
            print("=====================================")
            print("=====================================")
            print("=====================================")
            print("=====================================")
            print("=====================================")
            print("=====================================")
            print("=====================================")

        n_mal = params.n_malicious_client
        self.server = ServerAvg(model=base_model, optimizer=base_optimizer, n_clients=params.n_clients,
                                chosen_rate=params.chosen_rate,
                                dataset=server_dataset, batch_size=params.batch_size, device=params.device)

        handcraft_trigger, distributed = self.params.handcraft_trigger, self.params.distributed_trigger
        # print("handcraft_trigger:", handcraft_trigger, "distributed_trigger:", distributed)
        self.synthesizer = PatternSynthesizer(self.task, handcraft_trigger, distributed, (0, n_mal))
        self.attacks = Attacks(params, self.synthesizer)

        self.clients = list()
        malicious_ids = np.random.choice(range(params.n_clients), params.n_malicious_client, replace=False)
        self.malicious_ids = malicious_ids
        i_mal = 0

        for c in range(params.n_clients):
            sample_ids = splited_dataset[c]
            dataset = Subset(self.task.train_dataset, sample_ids)
            client_model = self.task.build_model()
            client_optimizer = self.task.build_optimizer(client_model)
            is_malicious = True if c in malicious_ids else False
            synthesizer, attacks = None, None
            if is_malicious:
                i_mal = i_mal + 1
                synthesizer = PatternSynthesizer(self.task, handcraft=handcraft_trigger, distributed=distributed, mal=(i_mal, n_mal))
                attacks = Attacks(params, synthesizer)

            client = Client(model=client_model, client_id=c, optimizer=client_optimizer, is_malicious=is_malicious,
                            dataset=dataset, local_epoch=params.local_epoch, batch_size=params.batch_size,
                            attacks=attacks, device=params.device)
            
            self.clients.append(client)
            print('build client:{} mal:{} data_num:{}'.format(c, is_malicious, len(dataset)))

    def fedavg_training(self, identifier=None):
        print("======================FedAvg Training =====================")
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: FedAvg Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)
            self.server.aggregate_global_model(self.clients, chosen_ids, None)
            print('Round {}: FedAvg Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False))
            fl_report.record_round_vars(self.test(epoch, backdoor=True))
            if (epoch + 1) % 1 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 30)

    def finetuning_training(self, identifier=None):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Finetuning Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)
            self.server.aggregate_global_model(self.clients, chosen_ids, None)
            print('Round {}: FedAvg Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            self.server.fine_tuning(self.task, self.clients, chosen_ids)
            print('Round {}: Finetuning Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': True})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': True})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 50)

    def mitigation_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Mitigation Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)

            self.server.aggregate_global_model(self.clients, chosen_ids, None)
            prune_order = self.server.collect_conv_ranks(self.task, self.clients, chosen_ids, None)
            if epoch % 5 == 0:
                self.server.conv_pruning(self.task, orders=prune_order)
                self.server.adjust_extreme_parameters(threshold=3)
            
            print('Round {}: Mitigation Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})


    def ensemble_distillation_training(self, identifier=None):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Ensemble Distillation Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)
            self.server.aggregate_global_model(self.clients, chosen_ids, None)
            print('Round {}: FedAvg Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            self.server.ensemble_distillation(self.task, self.clients, chosen_ids)
            print('Round {}: Distillation Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': True})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': True})
            if (epoch + 1) % 10 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 50)

    def adaptive_distillation_training(self, identifier=None):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Adaptive Distillation Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)
            pts = self.server.get_median_scores(self.task, self.clients, chosen_ids)
            self.server.aggregate_global_model(self.clients, chosen_ids, pts)

            print('Round {}: FedAvg Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            self.server.adaptive_distillation(self.task, self.clients, chosen_ids)
            print('Round {}: FedRAD Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': True})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': True})
            if (epoch + 1) % 10 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 50)

    def crfl_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: CRFL Distillation Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)

            self.server.aggregate_global_model(self.clients, chosen_ids, None)
            # print('Round {}: FedAvg Testing'.format(epoch))
            # fl_report.record_round_vars(self.crfl_test(epoch, backdoor=False), notation={'is_distill': False})
            # fl_report.record_round_vars(self.crfl_test(epoch, backdoor=True), notation={'is_distill': False})

            if not epoch == self.params.n_epochs - 1:
                self.server.clip_weight_norm()
                self.server.add_differential_privacy_noise(sigma=0.002, cp=False)
            print('Round {}: CRFL Testing'.format(epoch))
            fl_report.record_round_vars(self.crfl_test(epoch, backdoor=False), notation={'is_distill': True})
            fl_report.record_round_vars(self.crfl_test(epoch, backdoor=True), notation={'is_distill': True})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
                
    def deepsight_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Deep-Sight Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)

            self.server.deepsight_aggregate_global_model(self.clients, chosen_ids, self.task, None)
            print('Round {}: Deep-Sight Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
                
    def robust_lr_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Robust LR Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)

            flip_analysis = self.server.sign_voting_aggregate_global_model(self.clients, chosen_ids, None)
            print('Round {}: Robust LR Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation=None)
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'flip_analysis': flip_analysis})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))

    def bulyan_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Bulyan Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)

            self.server.bulyan_aggregate_global_model(self.clients, chosen_ids, None)
            print('Round {}: Bulyan Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))

    def backdoor_unlearning_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Backdoor Unlearning Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)
            
            self.server.aggregate_global_model(self.clients, chosen_ids, None)

    def flame_training(self, identifier):
        print("======================Flame Training =====================")
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)
        all_candidates = []
        for client in self.clients:
            all_candidates.append(client.client_id)
        for epoch in range(self.params.n_epochs):
            print('Round {}: Backdoor Unlearning Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            print("server broadcast done")
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            print("chosen id done")
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    print("client idle")
                    client.idle()
                    print("client idle done")
                else:
                    print("client handcraft")
                    client.handcraft(self.task)
                    print("client handcraft done")
                    client.train(self.task)
                    print("client train done")
            print("client train done")
            # start flame defnese
            # first step: DBSCAN clustering
            # second step: normalize the model and clip the weights
            # third step: add noise to the model

            w_locals_k = []
            cos_list = []
            clients_weight_ = []
            clients_weight_total = []
            for i in range(self.params.n_clients):
                print("client:", i)
                # 将clients[i].local_model一维化
                client_weight = torch.tensor([])
                client_weight_total = torch.tensor([])
                for name, param in self.clients[i].local_model.state_dict():
                    client_weight = torch.cat((client_weight, param.reshape(-1).cpu()))
                    if name == 'fc.weight' or name == 'fc.bias':
                        client_weight_total = torch.cat((client_weight_total, (param + self.server.global_model.state_dict()[name]).reshape(-1).cpu()))

                clients_weight_.append(client_weight)
                clients_weight_total.append(client_weight_total)
            clients_weight_ = torch.stack(clients_weight_)
            clients_weight_total = torch.stack(clients_weight_total)
            num_clients = clients_weight_total.shape[0]
            euclidean = (clients_weight_ ** 2).sum(1).sqrt()
            med = euclidean.median()
            tpr, tnr = 0, 0
            cluster = hdbscan.HDBSCAN(min_cluster_size=self.params.n_clients // 2 + 1, max_cluster_size=int(self.params.n_clients * 3 / 4), min_samples=1, allow_single_cluster=True)
            cluster.fit(clients_weight_total)
            predict_good = []
            predict_bad = []
            for i, j in enumerate(cluster.labels_):
                if j == 0:
                    predict_good.append(self.clients[i])
                else:
                    predict_bad.append(self.clients[i])

            print(cluster.labels_)
            # predict_good = set(predict_good)
            # predict_bad = set(predict_bad)
            # true_bad = set(true_bad)
            # true_good = set(true_good)
            # if len(true_good) == 0 and len(predict_good) == 0:
            #     tnr = 1
            # elif len(predict_good) == 0 and len(true_good) != 0:
            #     tnr = 0
            # else:
            #     tnr = len(true_good & predict_good) / len(predict_good)
            #
            # if len(true_bad) == 0 and len(predict_bad) == 0:
            #     tpr = 1
            # elif len(predict_bad) == 0 and len(true_bad) != 0:
            #     tpr = 0
            # else:
            #     tpr = len(true_bad & predict_bad) / len(predict_bad)
            # 2. 范数中值裁剪

            for i in range(self.params.n_clients):
                gama = med.div(euclidean[i])
                if gama > 1:
                    gama = 1

                for name, params in self.clients[i].local_model:
                    params.data = (params.data * gama).to(params.data.dtype)

            # self.server.global_model.load_state_dict(w_glob_k)
            # self.server.aggregate_global_model_with_weights(predict_good)
            self.server.aggregate_global_model(predict_good, chosen_ids, None)
            print('Round {}: Flame Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            if (epoch + 1) % 5 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))








    def test(self, epoch, backdoor, another_model=None):
        if self.params.handcraft and self.params.handcraft_trigger:
            self.attacks.synthesizer.pattern = 0
            for i in self.malicious_ids:
                self.attacks.synthesizer.pattern += self.clients[i].attacks.synthesizer.pattern
            self.attacks.synthesizer.pattern = self.attacks.synthesizer.pattern / len(self.malicious_ids)

        target_model = self.server.global_model if another_model is None else another_model
        target_model.eval()
        test_loader = self.task.test_loader
        for metric in self.task.metrics:
            metric.reset_metric()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                batch = self.task.get_batch(i, data)
                if backdoor:
                    batch = self.attacks.synthesizer.make_backdoor_batch(batch, test=True, attack=True)

                outputs = target_model(batch.inputs)
                '''To Modify'''
                self.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
            print("backdoor:{} metric:{}".format(backdoor, self.task.metrics))

        round_info = dict()
        for metric in self.task.metrics:
            round_info.update(metric.get_value())
        round_info['backdoor'] = backdoor
        round_info['epoch'] = epoch
        return round_info

    def crfl_test(self, epoch, backdoor, another_model=None):
        if self.params.handcraft:
            self.attacks.synthesizer.pattern = 0
            for i in self.malicious_ids:
                self.attacks.synthesizer.pattern += self.clients[i].attacks.synthesizer.pattern
            self.attacks.synthesizer.pattern = self.attacks.synthesizer.pattern / len(self.malicious_ids)

        target_model = self.server.global_model if another_model is None else another_model
        target_model.eval()
        test_loader = self.task.test_loader

        smoothed_models = [self.server.add_differential_privacy_noise(sigma=0.002, cp=True) for m in range(5)]
        for metric in self.task.metrics:
            metric.reset_metric()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                batch = self.task.get_batch(i, data)
                if backdoor:
                    batch = self.attacks.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
                outputs = 0
                for target_model in smoothed_models:
                    prob = torch.nn.functional.softmax(target_model(batch.inputs), 1)
                    # print("prob:",prob[0])
                    outputs = outputs + prob
                outputs = outputs / (len(smoothed_models))
                self.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
            print("backdoor:{} metric:{}".format(backdoor, self.task.metrics))

        round_info = dict()
        for metric in self.task.metrics:
            round_info.update(metric.get_value())
        round_info['backdoor'] = backdoor
        round_info['epoch'] = epoch
        return round_info

    def make_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        return nn.CrossEntropyLoss(reduction='none')

    def generate_malicious_client_ids(self, n_client, n_malicious_client, test):
        if test:
            return [0]
        else:
            return np.random.choice(range(n_client), n_malicious_client, replace=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pass in a parameter')
    parser.add_argument('--defense', type=str, help='defence name')
    parser.add_argument('--config', type=str, help='configs', choices=['cifar', 'imagenet'])
    parser.add_argument('--backdoor', type=str, help='type of backdoor attacks',
                        choices=['neurotoxin', 'ff', 'dba', 'naive', 'baseline'])
    parser.add_argument('--model', type=str, help='model', choices=['simple', 'resnet18'])
    args = parser.parse_args()
    configs = 'configs/{}_fed.yaml'.format(args.config)
    with open(configs) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params = Params(**params)
    params.defense = args.defense
    params.model = args.model
    
    
    # print("args backdoor:{}".format(args.backdoor))
    params.backdoor = args.backdoor
    if args.backdoor == 'ff':
        params.distributed_trigger = False
        params.handcraft = True
        params.handcraft_trigger = True
    elif args.backdoor == 'dba':
        params.distributed_trigger = True
        params.handcraft = False
        params.handcraft_trigger = False
    elif args.backdoor == 'naive':
        params.distributed_trigger = False
        params.handcraft = False
        params.handcraft_trigger = False
    elif args.backdoor == 'neurotoxin':
        params.distributed_trigger = False
        params.handcraft = False
        params.handcraft_trigger = False
    else:
        print("Not implemented defenses")

    fl_report = FLReport()
    experiment_name = "{}/{}_{}_{}_{}_h{}_c{}".format(params.resultdir, args.backdoor, args.defense, args.config,
                                                      args.model, params.heterogenuity, params.n_clients)
    experiment = FederatedBackdoorExperiment(params)

    if params.defense == 'fedavg':
        experiment.fedavg_training(identifier=experiment_name)
    elif params.defense == 'ensemble-distillation':
        experiment.ensemble_distillation_training(identifier=experiment_name)
    elif params.defense == 'mediod-distillation':
        experiment.adaptive_distillation_training(identifier=experiment_name)
    elif params.defense == 'fine-tuning':
        experiment.finetuning_training(identifier=experiment_name)
    elif params.defense == 'mitigation-pruning':
        experiment.mitigation_training(identifier=experiment_name)
    elif params.defense == 'robustlr':
        experiment.robust_lr_training(identifier=experiment_name)
    elif params.defense == 'certified-robustness':
        experiment.crfl_training(identifier=experiment_name)
    elif params.defense == 'bulyan':
        experiment.bulyan_training(identifier=experiment_name)
    elif params.defense == 'flame':
        experiment.flame_training(identifier=experiment_name)
    elif params.defense == 'deep-sight':
        experiment.deepsight_training(identifier=experiment_name)
        print("Defence Name Errors")
