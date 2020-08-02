from .trainers import KnowledgeTrainer
from .trainers import PatternTrainer
from .trainers import DENetTrainer


def get_trainer(name):
    """get_loader"""
    print("Using trainer: {}".format(name))
    return key2trainer[name]


key2trainer = {
    'KnowledgeTrainer': KnowledgeTrainer,
    'PatternTrainer': PatternTrainer,
    'DENetTrainer': DENetTrainer,
}
