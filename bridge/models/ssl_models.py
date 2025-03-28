from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
import lightly
from lightly.loss import (
    NTXentLoss,
    NegativeCosineSimilarity,
    DINOLoss,
    BarlowTwinsLoss,
)
from lightly.models import utils
from lightly.models.modules import heads
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from transformers import AutoModel, AutoConfig


class AugLossWrapper(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, tup, *args, **kwargs):
        return self.loss_fn(*tup)


class LightlySSLModel(nn.Module):
    """Base class for Lightly SSL models"""

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 256,
        num_ftrs: Optional[int] = None,
        pretrained: bool = False,
        transform: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.projection_dim = projection_dim

        # Create backbone
        self.backbone, self.num_ftrs = self._create_backbone(
            backbone, pretrained=pretrained
        )

        if num_ftrs is not None:
            self.num_ftrs = num_ftrs

    def _create_backbone(
        self, backbone: str, pretrained: bool = True
    ) -> Tuple[nn.Module, int]:
        """Create the backbone using HuggingFace transformers and return it with the number of features"""
        # Load pre-trained model if pretrained=True, otherwise initialize with random weights
        config = AutoConfig.from_pretrained(backbone)

        if pretrained:
            model = AutoModel.from_pretrained(backbone)
        else:
            model = AutoModel.from_config(config)

        # Get the number of features from the model's configuration
        # Different model architectures store this information differently
        if hasattr(config, "hidden_size"):
            # For BERT, ViT, etc.
            num_ftrs = config.hidden_size
        elif hasattr(config, "hidden_dim"):
            # Some vision models
            num_ftrs = config.hidden_dim
        elif hasattr(config, "num_channels"):
            # For ResNet-like models from HF
            # This would need a pooling layer typically
            num_ftrs = config.hidden_sizes[-1] # Last channels count
        else:
            raise ValueError(
                f"Could not determine feature dimension for model {backbone}"
            )

        # For vision models, we need a feature extractor
        # Modify the model to return the features before the classification head
        if "resnet" in backbone.lower() or "vit" in backbone.lower():
            # Create a wrapper module for vision models
            class FeatureExtractor(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    outputs = self.model(x, output_hidden_states=True)
                    if hasattr(outputs, "pooler_output"):
                        return outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):
                        # For ViT, use cls token
                        return outputs.last_hidden_state[:, 0]
                    else:
                        # For ResNet, use the final feature map (needs pooling)
                        features = outputs.hidden_states[-1]
                        return torch.mean(
                            features, dim=[2, 3]
                        )  # Global average pooling

            model = FeatureExtractor(model)

        return model, num_ftrs

    def forward(self, x):
        """Extract features from the backbone"""
        return self.backbone(x)

    def get_features(self, x):
        """Extract features from the backbone"""
        return self.forward(x)


class DINO(LightlySSLModel):
    """Implementation of DINO self-supervised learning method"""

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 256,
        pretrained: bool = False,
        use_momentum_encoder: bool = True,
        momentum_tau: float = 0.996,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
        optimizer: dict = {},
        *args,
        **kwargs,
    ):
        super().__init__(
            backbone, projection_dim, pretrained=pretrained, transform=DINOTransform()
        )

        # Create model using lightly's DINO implementation
        self.model = lightly.models.DINO(
            self.backbone,
            num_ftrs=self.num_ftrs,
            out_dim=projection_dim,
            use_momentum_encoder=use_momentum_encoder,
            m=momentum_tau,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )

        self.loss_fn = DINOLoss(
            output_dim=self.projection_dim,
            warmup_teacher_temp_epochs=5,
        )
        
        self.optimizer = Adam

        self.transform = DINOTransform()

    def forward(self, x):
        """Extract features from the backbone"""
        if self.training:
            # During training, return the full DINO model output
            return self.model(x)
        else:
            # During inference, return only the backbone features
            return self.model.backbone(x)

    def get_features(self, x):
        """Extract features from the backbone (for inference)"""
        return self.model.backbone(x)


class SimCLR(LightlySSLModel):
    """Implementation of SimCLR self-supervised learning method"""

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 128,
        pretrained: bool = False,
        temperature: float = 0.5,
        optimizer: dict = {},
        *args,
        **kwargs,
    ):
        super().__init__(
            backbone, projection_dim, pretrained=pretrained, transform=SimCLRTransform()
        )

        # Create model using lightly's SimCLR implementation
        self.model = lightly.models.SimCLR(
            self.backbone,
            num_ftrs=self.num_ftrs,
            out_dim=projection_dim,
        )

        # Store temperature for loss calculation
        self.temperature = temperature

        self.loss_fn = AugLossWrapper(NTXentLoss(temperature=self.temperature))

        self.optimizer = Adam

        self.transform = SimCLRTransform()

    def forward(self, x):
        x = self.transform(x)

        if self.training:
            # During training, return the full SimCLR model output
            return self.model(*x)
        else:
            # During inference, return only the backbone features
            return self.model.backbone(x)

    def get_features(self, x):
        """Extract features from the backbone (for inference)"""
        return self.model.backbone(x)


class BarlowTwins(LightlySSLModel):
    """Implementation of Barlow Twins self-supervised learning method"""

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 2048,
        pretrained: bool = False,
        lambda_param: float = 0.0051,
        input_size: int = 224,
        optimizer: dict = {},
        *args,
        **kwargs,
    ):
        super().__init__(
            backbone,
            projection_dim,
            pretrained=pretrained,
            transform=BYOLTransform(
                view_1_transform=BYOLView1Transform(
                    input_size=input_size, gaussian_blur=0.0
                ),
                view_2_transform=BYOLView2Transform(
                    input_size=input_size, gaussian_blur=0.0
                ),
            ),
        )

        # Create model using lightly's BarlowTwins implementation
        self.model = lightly.models.BarlowTwins(
            self.backbone,
            num_ftrs=self.num_ftrs,
            out_dim=projection_dim,
        )

        # Store lambda parameter for loss calculation
        self.lambda_param = lambda_param
    
        self.loss_fn = BarlowTwinsLoss(lambda_param=self.lambda_param)
      
        self.optimizer = Adam

        self.transform = BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=224, gaussian_blur=0.0),
            view_2_transform=BYOLView2Transform(input_size=224, gaussian_blur=0.0),
        )

    def forward(self, x):
        if self.training:
            # During training, return the full BarlowTwins model output
            return self.model(x)
        else:
            # During inference, return only the backbone features
            return self.model.backbone(x)

    def get_features(self, x):
        """Extract features from the backbone (for inference)"""
        return self.model.backbone(x)


class BYOL(LightlySSLModel):
    """Implementation of BYOL self-supervised learning method"""

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 128,
        pretrained: bool = False,
        momentum_tau: float = 0.996,
        input_size: int = 224,
        optimizer: dict = {},
        *args,
        **kwargs,
    ):
        super().__init__(
            backbone,
            projection_dim,
            pretrained=pretrained,
            transform=BYOLTransform(
                view_1_transform=BYOLView1Transform(
                    input_size=input_size, gaussian_blur=0.0
                ),
                view_2_transform=BYOLView2Transform(
                    input_size=input_size, gaussian_blur=0.0
                ),
            ),
        )

        # Create model using lightly's BYOL implementation
        self.model = lightly.models.BYOL(
            self.backbone,
            num_ftrs=self.num_ftrs,
            out_dim=projection_dim,
            m=momentum_tau,
        )

        self.loss_fn = NegativeCosineSimilarity()

        self.optimizer = Adam

        self.transform = BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=224, gaussian_blur=0.0),
            view_2_transform=BYOLView2Transform(input_size=224, gaussian_blur=0.0),
        )

    def forward(self, x):
        if self.training:
            # During training, return the full BYOL model output
            return self.model(x)
        else:
            # During inference, return only the backbone features
            return self.model.backbone(x)

    def get_features(self, x):
        """Extract features from the backbone (for inference)"""
        return self.model.backbone(x)


class MoCo(LightlySSLModel):
    """Implementation of MoCo v1/v2 self-supervised learning method"""

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 128,
        pretrained: bool = False,
        momentum_tau: float = 0.996,
        temperature: float = 0.1,
        queue_size: int = 65536,
        use_momentum_encoder: bool = True,
        optimizer: dict = {},
        *args,
        **kwargs,
    ):
        super().__init__(
            backbone, projection_dim, pretrained=pretrained, transform=MoCoV2Transform()
        )

        # Create model using lightly's MoCo implementation
        self.model = lightly.models.MoCo(
            self.backbone,
            num_ftrs=self.num_ftrs,
            out_dim=projection_dim,
            m=momentum_tau,
            temperature=temperature,
            queue_size=queue_size,
        )

        self.loss_fn = NTXentLoss(memory_bank_size=self.model.queue_size)

        self.optimizer = Larc

        self.transform = MoCoV2Transform()

    def forward(self, x):
        if self.training:
            # During training, return the full MoCo model output
            return self.model(x)
        else:
            # During inference, return only the backbone features
            return self.model.backbone(x)

    def get_features(self, x):
        """Extract features from the backbone (for inference)"""
        return self.model.backbone(x)


class MoCoV3(LightlySSLModel):
    """Implementation of MoCo v3 self-supervised learning method"""

    def __init__(
        self,
        backbone: str = "vit_small",
        projection_dim: int = 256,
        pretrained: bool = False,
        momentum_tau: float = 0.996,
        temperature: float = 0.2,
        use_momentum_encoder: bool = True,
        proj_hidden_dim: int = 4096,
        pred_hidden_dim: int = 4096,
        use_predictor: bool = True,
        use_queue: bool = False,
        queue_size: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(
            backbone, projection_dim, pretrained=pretrained, transform=MocoV2Transform()
        )

        # Create model using lightly's MoCoV3 implementation
        # In lightly, MoCoV3 is built with custom projector and predictor
        self.model = lightly.models.MoCoV3(
            self.backbone,
            num_ftrs=self.num_ftrs,
            proj_hidden_dim=proj_hidden_dim,
            pred_hidden_dim=pred_hidden_dim,
            out_dim=projection_dim,
            m=momentum_tau,
            temperature=temperature,
            use_predictor=use_predictor,
        )

        self.loss_fn = NTXentLoss(memory_bank_size=self.model.queue_size)

        self.transform = MoCoV2Transform()

    def forward(self, x):
        if self.training:
            # During training, return the full MoCoV3 model output
            return self.model(x)
        else:
            # During inference, return only the backbone features
            return self.model.backbone(x)

    def get_features(self, x):
        """Extract features from the backbone (for inference)"""
        return self.model.backbone(x)


# Factory function to get model by name
def get_ssl_model(name: str, **kwargs) -> LightlySSLModel:
    """Get SSL model by name"""
    models = {
        "dino": DINO,
        "simclr": SimCLR,
        "barlow_twins": BarlowTwins,
        "byol": BYOL,
        "moco": MoCo,
        "mocov3": MoCoV3,
    }

    if name.lower() not in models:
        raise ValueError(
            f"Unknown SSL model: {name}. Available models: {list(models.keys())}"
        )

    return models[name.lower()](**kwargs)
