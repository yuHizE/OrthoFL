from transformers import AutoModelForSequenceClassification
import torch.nn as nn

from models.lenet import LeNet5
from models.vgg import VGG
from models.resnet1d import ResNet1D
from models.mobilenetv2 import MobileNetV2

def create_peft_config(base_model):
    target_modules = []
    modules_to_save = []
    for n, m in base_model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            target_modules.append(n)
        elif isinstance(m, nn.BatchNorm2d):
            modules_to_save.append(n)

    return target_modules, modules_to_save

def build_model(model_name, task, n_class, peft_config=None, base_model_name=None):
    if task.startswith('cifar'):
        input_size = 3
    elif task == 'mnist':
        input_size = 1
    elif task == 'har':
        input_size = 9

    if model_name.endswith('_lora'):
        from peft import get_peft_model
        # Define the model configuration
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=n_class)
        model = get_peft_model(base_model, peft_config)
    elif model_name == 'MobileNetV2':
        model = MobileNetV2(n_class)
    elif model_name == 'ResNet1D': # for time-series
        model = ResNet1D(input_size, n_class)
    elif model_name == 'LeNet5':
        model = LeNet5(n_class, input_size)
    elif model_name == 'VGG11':
        model = VGG('VGG11')
    elif model_name == 'DistilBERT':
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=n_class)

    else:
        raise ValueError('Wrong model name:', model_name)

    return model

