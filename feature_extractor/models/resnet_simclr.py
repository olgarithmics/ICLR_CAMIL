import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from collections import OrderedDict

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.features = nn.Sequential(*list(feature_extractor.children())[:-1])
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.features(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)}

        resnet = self._get_basemodel(base_model)

        for param in resnet.parameters():
            param.requires_grad = False

        for param in resnet.layer4.parameters():
            param.requires_grad = True

        num_ftrs = resnet.fc.in_features

        self.i_classifier = IClassifier(resnet, num_ftrs, output_class=num_ftrs).cuda()

        self.i_classifier = self.load_model_weights(self.i_classifier,'runs/tcga_lung/checkpoints/model-v0.pth')

        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")


    def load_model_weights(self, model, weights):
        state_dict_weights = torch.load(weights)
        state_dict_init = model.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print ('Loading weights from checkpoint', flush=True)
        return model

    def forward(self, x):
        feats, h = self.i_classifier(x)
        x = F.relu(h)
        x = self.l2(x)
        return feats, x
