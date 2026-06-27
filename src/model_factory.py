from src.model_attention_unet import AttentionUNet
from src.model_segformer import SegFormerBinary
from src.model_unet import UNet
from src.utils import canonical_model_name


def get_model(model_name, in_channels=3, out_channels=1, **kwargs):
    name = canonical_model_name(model_name)
    base_channels = int(kwargs.get("base_channels", 32))

    if name == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name == "attention_unet":
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name == "segformer":
        return SegFormerBinary(
            encoder_name=kwargs.get("encoder_name", "nvidia/mit-b2"),
            encoder_weights=kwargs.get("encoder_weights", "imagenet"),
            in_channels=in_channels,
            out_channels=out_channels,
        )

    third_party = {
        "unet_plus_plus": "UnetPlusPlus",
        "unetplusplus": "UnetPlusPlus",
        "deeplabv3plus": "DeepLabV3Plus",
        "fpn": "FPN",
        "manet": "MAnet",
    }
    if name in third_party:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                f"Model `{model_name}` requires segmentation-models-pytorch. "
                "Install it with `pip install segmentation-models-pytorch`."
            ) from exc
        cls = getattr(smp, third_party[name])
        return cls(
            encoder_name=kwargs.get("encoder_name", "resnet34"),
            encoder_weights=kwargs.get("encoder_weights", "imagenet"),
            in_channels=in_channels,
            classes=out_channels,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")
