import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


def visualization_data(cluster_model, transformer, decomposition_model, data, target, metrics_dict, handler,
                       method_name):
    data = decomposition_model.fit_transform(data)
    if transformer is not None:
        data = transformer.transform(data)

    visual_model = TruncatedSVD(n_components=2)
    projection = visual_model.fit_transform(data)

    cluster_model.fit(data)
    predict_target = cluster_model.labels_

    map_color = {0: 'r', 1: 'g'}

    def generate_color(label_list):
        return list(map(lambda x: map_color[x], label_list))

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(projection[:, 0], projection[:, 1], c='', marker='o', edgecolors=generate_color(predict_target))
    ax[0].set_title('Cluster result')
    ax[1].scatter(projection[:, 0], projection[:, 1], c='', marker='o', edgecolors=generate_color(target))
    ax[1].set_title('Ground Truth')
    plt.savefig('Result/'+method_name + '_cluster.jpg')

    handler.write('\n')
    handler.write('*' * 20 + method_name + '*' * 20 + '\n')
    for metrics_name in metrics_dict.keys():
        handler.write(metrics_name + ': {: 0.4f}\n'.format(metrics_dict[metrics_name](target, predict_target)))


