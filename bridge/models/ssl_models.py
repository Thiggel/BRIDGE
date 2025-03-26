from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
import lightly
from lightly.models import utils
from lightly.models.modules import heads


class LightlySSLModel(nn.Module):
    """Base class for Lightly SSL models"""

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 256,
        num_ftrs: Optional[int] = None,
        pretrained: bool = False,
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
        self, backbone: str, pretrained: bool = False
    ) -> Tuple[nn.Module, int]:
        """Create the backbone and return it with the number of features"""
        backbone_model, num_ftrs = utils.get_backbone(backbone, pretrained=pretrained)
        return backbone_model, num_ftrs

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
    ):
        super().__init__(backbone, projection_dim, pretrained=pretrained)

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
    ):
        super().__init__(backbone, projection_dim, pretrained=pretrained)

        # Create model using lightly's SimCLR implementation
        self.model = lightly.models.SimCLR(
            self.backbone,
            num_ftrs=self.num_ftrs,
            out_dim=projection_dim,
        )

        # Store temperature for loss calculation
        self.temperature = temperature

    def forward(self, x):
        if self.training:
            # During training, return the full SimCLR model output
            return self.model(x)
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
    ):
        super().__init__(backbone, projection_dim, pretrained=pretrained)

        # Create model using lightly's BarlowTwins implementation
        self.model = lightly.models.BarlowTwins(
            self.backbone,
            num_ftrs=self.num_ftrs,
            out_dim=projection_dim,
        )

        # Store lambda parameter for loss calculation
        self.lambda_param = lambda_param

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
    ):
        super().__init__(backbone, projection_dim, pretrained=pretrained)

        # Create model using lightly's BYOL implementation
        self.model = lightly.models.BYOL(
            self.backbone,
            num_ftrs=self.num_ftrs,
            out_dim=projection_dim,
            m=momentum_tau,
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
    ):
        super().__init__(backbone, projection_dim, pretrained=pretrained)

        # Create model using lightly's MoCo implementation
        self.model = lightly.models.MoCo(
            self.backbone,
            num_ftrs=self.num_ftrs,
            out_dim=projection_dim,
            m=momentum_tau,
            temperature=temperature,
            queue_size=queue_size,
        )

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
    ):
        super().__init__(backbone, projection_dim, pretrained=pretrained)

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
