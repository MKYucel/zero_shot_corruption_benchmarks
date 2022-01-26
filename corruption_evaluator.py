import torch
import numpy as np
from tools import harmonic_score_gzsl, normalized_accuracy_zsl
import torch.optim as optim
from art.classifiers import PyTorchClassifier
import torchvision
import torch.nn as nn
from fullgraph import FullGraph
import time

def zsl_launch(dataloader, unseenVectors, criterion, params):
    if params["dataset"] == "CUB":
        from configs.config_CUB import cub_model_paths
        model_path = cub_model_paths[params["test_model"]]
    elif params["dataset"] == "AWA2":
        from configs.config_AWA2 import awa_model_paths
        model_path = awa_model_paths[params["test_model"]]
    elif params["dataset"] == "SUN":
        from configs.config_SUN import sun_model_paths
        model_path = sun_model_paths[params["test_model"]]

    resnet = torchvision.models.resnet101(pretrained=True).cuda()
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    print("Loading:", model_path)
    model_ale = torch.load(model_path).cuda()
    if params["test_model"] == "ant" or params["test_model"] == "augmix":
        model_ale = model_ale.ale_graph

    full_graph = FullGraph(feature_extractor, model_ale, unseenVectors).cuda()
    full_graph.eval()
    optimizer = optim.SGD(full_graph.parameters(), lr=0.01, momentum=0.5)

    if params["dataset"] == "CUB":
        no_classes = 50
    elif params["dataset"] == "AWA2":
        no_classes = 10
    elif params["dataset"] == "SUN":
        no_classes = 72

    classifier = PyTorchClassifier(model=full_graph,  loss=criterion,
                                   optimizer=optimizer, input_shape=(1, 150, 150), nb_classes=no_classes)
    batch_size = params["batch_size"]

    preds = []
    labels_ = []
    start= time.time()

    for index, sample in enumerate(dataloader):
        img = sample[0].numpy()
        label = sample[1].numpy()

        predictions = classifier.predict(img, batch_size=batch_size)
        preds.extend(np.argmax(predictions, axis=1))
        labels_.extend(label)

        if index % 1000 ==0:
            print(index, len(dataloader))

    end=time.time()

    labels_ = np.array(labels_)

    acc_adversarial  = normalized_accuracy_zsl(preds, labels_)

    print("ZSL Top-1:", acc_adversarial)
    print(end-start , "seconds passed for ZSL.")

def gzsl_launch(dataloader_seen, dataloader_unseen, all_vectors, criterion, params):
    if params["dataset"] == "CUB":
        from configs.config_CUB import cub_model_paths
        model_path = cub_model_paths[params["test_model"]]
    elif params["dataset"] == "AWA2":
        from configs.config_AWA2 import awa_model_paths
        model_path = awa_model_paths[params["test_model"]]
    elif params["dataset"] == "SUN":
        from configs.config_SUN import sun_model_paths
        model_path = sun_model_paths[params["test_model"]]

    resnet = torchvision.models.resnet101(pretrained=True).cuda()
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    print("Loading:", model_path)
    model_ale = torch.load(model_path).cuda()
    if params["test_model"] == "ant" or params["test_model"] == "augmix":
        model_ale = model_ale.ale_graph

    full_graph = FullGraph(feature_extractor, model_ale, all_vectors).cuda()
    full_graph.eval()
    optimizer = optim.SGD(full_graph.parameters(), lr=0.01, momentum=0.5)

    if params["dataset"] == "CUB":
        no_classes = 200
    elif params["dataset"] == "AWA2":
        no_classes = 50
    elif params["dataset"] == "SUN":
        no_classes = 717

    classifier = PyTorchClassifier(model=full_graph, loss=criterion,
                                   optimizer=optimizer, input_shape=(1, 150, 150), nb_classes=no_classes)
    batch_size = params["batch_size"]

    preds_seen = []
    labels_seen_ = []
    start= time.time()

    for index, sample in enumerate(dataloader_seen):
        img = sample[0].numpy()
        label = sample[1].numpy()

        predictions = classifier.predict(img, batch_size=batch_size)
        preds_seen.extend(np.argmax(predictions, axis=1))
        labels_seen_.extend(label)

        if index % 1000 ==0:
            print(index, len(dataloader_seen))

    labels_seen_ = np.array(labels_seen_)
    uniq_labels_seen = np.unique(labels_seen_)

    labels_unseen_ = []
    preds_unseen = []
    preds_seen = np.array(preds_seen)

    for index, sample in enumerate(dataloader_unseen):
        img = sample[0].numpy()
        label = sample[1].numpy()

        predictions = classifier.predict(img, batch_size=batch_size)
        preds_unseen.extend(np.argmax(predictions, axis=1))
        labels_unseen_.extend(label)

        if index % 1000 ==0:
            print(index, len(dataloader_unseen))

    end= time.time()

    labels_unseen_ = np.array(labels_unseen_)
    uniq_labels_unseen = np.unique(labels_unseen_)

    combined_labels = np.concatenate((labels_seen_, labels_unseen_))
    preds_unseen = np.array(preds_unseen)
    combined_preds = np.concatenate((preds_seen, preds_unseen))
    harmonic_score_gzsl(combined_preds, combined_labels, uniq_labels_seen, uniq_labels_unseen)

    print(end-start , "seconds passed for GZSL.")
