
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def construct_graph(dataset, features, topk):
    fname = '../data/' + dataset + '/knn/tmp.txt'
    print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn(dataset):

        topk = 2
        data = np.loadtxt('../data/' + dataset + '/' + dataset + '.feature', dtype=float)
        # print(data)
        construct_graph(dataset, data, topk)
        f1 = open('../data/' + dataset + '/knn/tmp.txt', 'r')
        f2 = open('../data/' + dataset + '/knn/c'+str(topk)+'.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            print(line)
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{}\t{}\n'.format(start, end))
        f2.close()

'''generate KNN graph'''

# generate_knn('dti')
