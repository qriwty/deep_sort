from torchvision import transforms

from .feature_model import FeatureModel
from torchvision.models import resnet18


class ResNet18Config(FeatureModel):
    def __init__(self, weights_path=None, use_cuda=False):
        super(ResNet18Config, self).__init__(use_cuda=use_cuda)

        self.configure_model(weights_path)

    def configure_model(self, weights_path=None):
        self.model_base = resnet18(pretrained=True if not weights_path else False)
        self.model_base.to(self.device)

        if weights_path:
            self.load(weights_path)

        self.input_shape = (224, 224)

        self.feature_layer = 'avgpool'

        self._preprocessor = transforms.Compose([
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
