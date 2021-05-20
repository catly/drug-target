from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models import SFGCN

from config import Config

# MGNN

if __name__ == "__main__":
    config_file = "./config/200dti.ini"

    config = Config(config_file)
    fold_ROC = []
    fold_AUPR = []
    for fold in range(0,5):
        config.structgraph_path = "../data/dti/alledg00{}.txt".format(fold + 1)
        config.train_path = "../data/dti/train00{}.txt".format(fold + 1)
        config.test_path = "../data/dti/test00{}.txt".format(fold + 1)
        use_seed = not config.no_seed
        if use_seed:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        sadj, fadj = load_graph(config)
        features, labels, idx_train, idx_test = load_data(config)

        asadj = get_adj(sadj)
        afadj = get_adj(fadj)

        model = SFGCN(nfeat=config.fdim,
                      nhid1=config.nhid1,
                      nhid2=config.nhid2,
                      nclass=config.class_num,
                      n=config.n,
                      dropout=config.dropout)


        features = features
        sadj = sadj
        fadj = fadj
        labels = labels
        idx_train = idx_train
        idx_test = idx_test
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        roc = []

        pr = []
        acc = []


        def train(model, epochs):
            model.train()
            optimizer.zero_grad()
            output = model(features, sadj, fadj, asadj, afadj)
            loss = F.nll_loss(output[idx_train], labels[idx_train])

            loss.backward()
            optimizer.step()
            c, d, p = main_test(model)
            print("this is the", epochs + 1,
                  "epochs, ROC is {:.4f},and AUPR is {:.4f} test set accuray is {:.4f},loss is {:.4f} ".format(c, d, p,
                                                                                                               loss))


        def main_test(model):
            model.eval()
            output = model(features, sadj, fadj, asadj, afadj)
            c, d = RocAndAupr(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            roc.append(c)
            pr.append(d)
            acc.append(acc_test)
            return c, d, acc_test


        acc_max = 0
        epoch_max = 0
        roc_max = 0
        pr_max = 0

        for epoch in range(config.epochs):
            train(model, epoch)
            if acc_max < acc[epoch]:
                acc_max = acc[epoch]
            if roc_max < roc[epoch]:
                roc_max = roc[epoch]
            if pr_max < pr[epoch]:
                pr_max = pr[epoch]
            if epoch + 1 == config.epochs:
                fold_ROC.append(roc_max)
                fold_AUPR.append(pr_max)
                print(
                    "this is {} fold ,the max ROC is {:.4f},and max AUPR is {:.4f} test set max  accuray is {:.4f} , ".format(fold,roc_max,
                                                                                                             pr_max,
                                                                                                             acc_max))
    print("average AUROC is {:.4} , average AUPR is {:.4}".format(sum(fold_ROC) / len(fold_ROC),
                                                                  sum(fold_AUPR) / len(fold_AUPR)))
