# coding=utf-8
import torch.optim as optim
from torchsummary import summary
import torch
import numpy as np
from memory import Memory
import os
from utils import AverageLoss, evaluator, get_Mahalanobis_convariance, mahalanobis
import time
from checkpoint import load_checkpoint, save_checkpoint, copy_state_dict, Logger
import sys
# from test import show, scatter_plot
import torch.nn.functional as F
from torchvision import datasets, transforms
from Autoencoder import MemAE
from datasets import Hyperloader
import torch.utils.data as Data
from opt_trainer import opt_sk


def get_features(model, dataloader):
    feature_list = []
    cluster_list = []
    label_list = []
    index_list = []
    for i, (data, label, index) in enumerate(dataloader):
        with torch.no_grad():
            data = data.cuda()
            features, _ = model.encoder(data)
            clusters = model.cluster_layer(features)
            feature_list.append(features.squeeze().detach())
            cluster_list.append(clusters)
            label_list.append(label)
            index_list.append(index)
    features = torch.cat(feature_list, dim=0)
    clusters = torch.cat(cluster_list, dim=0)
    label = torch.cat(label_list, dim=0)
    index = torch.cat(index_list, dim=0)
    return features, clusters, index


def chocolate_experiment(run_times, args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # print(torch.cuda.is_available())
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    running_info = '{}-cake{}-{}'.format(run_times, args.num_clusters, args.dataset)
    model_path = os.path.join(args.dump_path, running_info)
    train_loader = Data.DataLoader(dataset=Hyperloader(args),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   # pin_memory=True,
                                   num_workers=args.num_work)
    test_loader = Data.DataLoader(dataset=Hyperloader(args),
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_work)


    learner = MemAE(args).cuda()
    # learner = torch.nn.DataParallel(learner, device_ids=range(torch.cuda.device_count())).cuda()
    # summary(E, (1, 32, 32))
    optimizer = optim.Adam(learner.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    memory = Memory(args, memory_init=None)
    load_checkpoint(learner, optimizer, memory, model_path, args)

    N = len(train_loader.dataset)
    # optimize_times = ((args.epochs + 1.0001) * N * (np.linspace(0, 1, args.nopts))[::-1]).tolist()
    # optimize_times = [(args.epochs + 10) * N] + optimize_times
    # print('We will optimize L at epochs:', [np.round(1.0 * t / N, 2) for t in optimize_times], flush=True)

    # init selflabels randomly
    selflabels = np.zeros(N, dtype=np.int32)
    for qq in range(N):
        selflabels[qq] = qq % args.num_clusters
    selflabels = np.random.permutation(selflabels)
    selflabels = torch.LongTensor(selflabels).cuda()

    auroc_buf = []
    prin_buf = []
    prout_buf = []
    # train
    print("train starting.........")
    for epoch in range(args.start_epoch, args.epochs):
        learner.train()
        contrastive_loss = AverageLoss()
        # regularize_loss = AverageLoss()
        cluster_loss = AverageLoss()
        read_mse_loss = AverageLoss()
        batch_time = AverageLoss()
        end = time.time()

        for i, (data, label, index) in enumerate(train_loader):
            data = data.cuda()
            loss, loss_dic = learner(data, index, selflabels, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            contrastive_loss.update(loss_dic[0])
            cluster_loss.update(loss_dic[1])
            # regularize_loss.update(loss_dic[2])
            read_mse_loss.update(loss_dic[2])
            batch_time.update(time.time() - end)
            end = time.time()
        features, clusters, index = get_features(learner, train_loader)
        selflabels = opt_sk(clusters, index, args)
        if epoch > args.warmup_epochs:
            learner.memory.write(features, clusters)
            learner.memory.erase_attention(clusters)

        print('Current data information:  \t{}'.format(running_info))
        print('Epoch: [{}][{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Contrastive Loss {:.5f}\t'
              'Cluster Loss {:.5f}\t'
              # 'Regularize Loss {:.5f}\t'
              'Read mse loss {:.5f}\t'
              .format(epoch+1, i + 1, len(train_loader),
                      batch_time.val, batch_time.avg,
                      contrastive_loss.avg,
                      cluster_loss.avg,
                      read_mse_loss.avg
                      ))

        if epoch % 10 == 9 and epoch >= args.warmup_epoch:
            save_checkpoint({
                'current_running_times': run_times,
                'dataset_name': args.dataset,
                'epoch': epoch + 1,
                'memory_item': memory.memory_item,
                'state_dict': learner.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filepath=model_path)

            learner.eval()
            train_features, train_clusters, label = get_features(learner, train_loader)
            test_features, test_clusters, _ = get_features(learner, test_loader)
            test_features = test_features.cpu()
            mean_train, inv_cov_train = get_Mahalanobis_convariance(train_features.cpu().numpy())
            m_distance = np.zeros(len(test_features))
            for i in range(len(test_features)):
                m_distance[i] = mahalanobis(test_features[i], mean_train, inv_cov_train)

            read_item = learner.memory.read(clusters)
            errors = 1 - torch.sum((read_item - test_features.numpy()) ** 2, dim=1)

            # scatter_plot(encodes1, label, epoch, args)
            # scatter_show(y_train, targets, encodes2, 2, epoch)
            # histogram_modified(errors[y_train == 1], errors[y_train==0], args, epoch)
            # show(targets, embeddings1, epoch, args)
            roc_auc, pr_auc_norm, pr_auc_anom =\
                evaluator(errors, label, "errors", args)
            roc_auc_m, pr_auc_norm_m, pr_auc_anom_m = \
                evaluator(m_distance, label, "mahalanobis", args)
            # roc_auc_i, pr_auc_norm_i, pr_auc_anom_i = \
            #     evaluator(sim_scores, y_train, run_times, dataset_name, p, c, "similarity", args)
            # roc_auc_d, pr_auc_norm_d, pr_auc_anom_d = \
            #         evaluator(dis, y_train, run_times, dataset_name, p, c, "nearest", args)
            # roc_auc, pr_auc_norm, pr_auc_anom = 0, 0, 0

            auroc_buf.append(roc_auc)
            prin_buf.append(pr_auc_norm)
            prout_buf.append(pr_auc_anom)
    final_auroc, final_pr_in, final_pr_out = 0, 0, 0
    if len(auroc_buf) > 0:
        auroc_result = np.hstack(auroc_buf)
        prin_result = np.hstack(prin_buf)
        prout_result = np.hstack(prout_buf)
        best_index = np.argmax(auroc_result)
        final_auroc = auroc_result[best_index]
        final_pr_in = prin_result[best_index]
        final_pr_out = prout_result[best_index]
    return [final_auroc, final_pr_in, final_pr_out]








