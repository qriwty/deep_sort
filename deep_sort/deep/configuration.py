from .feature_model import FeatureModel

from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50


class ResNetConfiguration(FeatureModel):
    def __init__(self, base="resnet18", weights_path=None, use_cuda=False):
        super(ResNetConfiguration, self).__init__(use_cuda=use_cuda)

        self._models = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50
        }

        self.weights_path = weights_path

        self.model_base = self._get_model_base(base)
        self.model_base.to(self.device)

        if weights_path:
            self.load(weights_path)

        self.input_shape = (224, 224)

        self.feature_layer = "avgpool"

        self._preprocessor = transforms.Compose([
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_model_base(self, base):
        if base not in self._models:
            raise ValueError(f"Model {base} is not supported. Choose from {list(self._models.keys())}.")

        model_base = self._models[base](pretrained=True if not self.weights_path else False)

        return model_base
