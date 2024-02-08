import numpy as np
import sacred
import scipy.optimize as optimize
import torch
import torch.nn.functional as functional

import calibration as uncertaintycalibration
import densenet, resnet
from dataloader import get_dataloader

# Crée une expérience "exp_calibration" (Pas nécessaire)
exp = sacred.Experiment("exp_calibration")

# Fonction pour charger un modèle pré-entraîné en fonction du type de modèle
def load_model(model_path, model_type):
    if model_type == "cifar10_resnet20":
        # Charge le modèle spécifique "cifar10_resnet20" en utilisant le hub torch
        try:
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            model = None
    elif "resnet" in model_path:
        # Charge le modèle ResNet en fonction du type spécifié et du chemin
        model = {
            'resnet20': resnet.resnet20,
            'resnet32': resnet.resnet32,
            'resnet44': resnet.resnet44,
            'resnet56': resnet.resnet56,
            'resnet110': resnet.resnet110,
        }[model_type]()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    elif "densenet" in model_path:
        # Charge le modèle DenseNet en fonction du type spécifié et du chemin
        model = {
            "densenet121": densenet.densenet121
        }[model_type]()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model

# Fonction pour calculer l'Erreur de Calibration Attendue (ECE) d'un modèle
def get_model_ece(logits, labels, num_classes, logits_are_proba=False):
    if not logits_are_proba:
        probas = torch.softmax(logits, dim=1)
    else:
        probas = logits
    return uncertaintycalibration.get_calibration_error(probas, labels, p=1, debias=False, mode='top-label')

# Fonction pour calculer le Score Brier d'un modèle
def get_brier_score(logits, labels):
    target_one_hot = torch.nn.functional.one_hot(labels)
    probs = torch.softmax(logits, dim=1)
    return torch.mean(torch.sum((probs - target_one_hot)**2, axis=1))

# Fonction pour calculer l'erreur de classification d'un modèle
def get_classification_error(logits, labels):
    output = torch.argmax(logits, dim=1)
    correct = (output == labels).sum().item()
    return 1.0 - correct / list(output.size())[0]

# Fonction pour trouver la température optimale pour la méthode "Temperature Scalling"
def find_optimal_temperature_temperature_scaling(logits, labels, T_min=0.01, T_max=10.0):
    def objective(T):
        probas = logits / T
        return functional.cross_entropy(probas, labels).item()
    
    res = optimize.minimize_scalar(objective, bounds=[T_min, T_max], method='bounded')
    return float(res.x)

# Fonction pour trouver la température optimale pour la consistance des attentes
def find_optimal_temperature_expectation_consistency(logits, labels, T_min=0.01, T_max=10.0):
    validation_error = get_classification_error(logits, labels)

    def objective(T):
        probas = torch.max(torch.softmax(logits / T, dim=1), dim=1)[0]
        return torch.mean(probas) - (1.0 - validation_error)

    res = optimize.root_scalar(objective, bracket=[T_min, T_max])
    return float(res.root)

# Fonction pour obtenir les logits et les labels à partir des données en utilisant un modèle donné
def get_logits_and_labels_from_data(model, data_loader):
    model.eval()
    labels_list, logits_list = [], []
    with torch.no_grad():
        for data, labels in data_loader:
            logits_list += model(data).tolist()
            labels_list += labels.tolist()
    return torch.FloatTensor(logits_list), torch.LongTensor(labels_list)

# Configuration pour l'expérience en utilisant sacred
@exp.config
def config():
    model_path = "default.bin"
    model_type = "cifar10_resnet20"  # Utilisez le modèle spécifique
    dataset_str = "cifar10"
    random_seed = 42
    train_ratio = 0.9

# Fonction principale pour l'expérience
@exp.automain
def main(model_path, model_type, dataset_str, random_seed, train_ratio):
    num_classes = {
        "cifar10": 10,
        "cifar100": 100,
        "svhn": 10
    }[dataset_str]

    # Obtient les chargeurs de données pour les ensembles d'entraînement, de validation et de test
    _, validation_loader, test_loader = get_dataloader(dataset_str, random_seed, train_ratio, batch_size_train=128, batch_size_validation=128, batch_size_test=128)
    # Charge le modèle spécifié
    model = load_model(model_path, model_type)

    # Obtient les logits et les labels des ensembles de validation et de test
    validation_logits, validation_labels = get_logits_and_labels_from_data(model, validation_loader)
    test_logits, test_labels = get_logits_and_labels_from_data(model, test_loader)

    # Affiche l'erreur de classification sur l'ensemble de test
    print(f'Erreur de classification : {get_classification_error(test_logits, test_labels)}') 

    # Trouve les températures optimales pour le Température scalling (TS) et la consistance des attentes (TEC)
    temperature_ts = find_optimal_temperature_temperature_scaling(validation_logits, validation_labels)
    temperature_ec = find_optimal_temperature_expectation_consistency(validation_logits, validation_labels)

    # Calcule l'ECE pour l'ensemble de test en utilisant différentes méthodes de calibration
    test_ece = get_model_ece(test_logits, test_labels, num_classes)
    test_ece_ts = get_model_ece(test_logits / temperature_ts, test_labels, num_classes)
    test_ece_ec = get_model_ece(test_logits / temperature_ec, test_labels, num_classes)
    
    # Affiche les températures et les résultats ECE
    print(f'Température de TS et EC : {temperature_ts}, {temperature_ec}')
    print(f'Test ECE (Non calibré, TS, EC) : {test_ece}, {test_ece_ts}, {test_ece_ec}')
    
    # Calcule les scores Brier pour l'ensemble de test en utilisant les différentes méthodes de calibration
    test_brier = get_brier_score(test_logits, test_labels)
    test_brier_ts = get_brier_score(test_logits / temperature_ts, test_labels)
    test_brier_ec = get_brier_score(test_logits / temperature_ec, test_labels)
    print(f'Test BS (Non calibré, TS, EC) : {test_brier}, {test_brier_ts}, {test_brier_ec}')
