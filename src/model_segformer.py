from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class SegFormerBinary(nn.Module):
    def __init__(self, encoder_name="nvidia/mit-b2", encoder_weights="imagenet", in_channels=3, out_channels=1):
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("The v1.5 SegFormer wrapper currently requires three-channel RGB input.")
        if int(out_channels) != 1:
            raise ValueError("The project requires one-channel binary segmentation logits.")
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError("SegFormer requires `transformers`. Install requirements-kaggle.txt.") from exc

        if encoder_weights in {None, "none", False}:
            normalized_name = str(encoder_name).lower().replace("_", "-")
            offline_configs = {
                "nvidia/mit-b2": [3, 4, 6, 3],
                "mit-b2": [3, 4, 6, 3],
                "nvidia/mit-b3": [3, 4, 18, 3],
                "mit-b3": [3, 4, 18, 3],
            }
            if normalized_name not in offline_configs:
                raise ValueError(
                    "Offline SegFormer checkpoint reconstruction supports only MiT-B2 and MiT-B3."
                )
            cfg = SegformerConfig(
                num_channels=3,
                num_labels=1,
                depths=offline_configs[normalized_name],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[64, 128, 320, 512],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                decoder_hidden_size=768,
                drop_path_rate=0.1,
            )
            self.model = SegformerForSemanticSegmentation(cfg)
        else:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                encoder_name,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )

    @property
    def encoder(self):
        return self.model.segformer

    def forward(self, inputs):
        logits = self.model(pixel_values=inputs).logits
        return F.interpolate(logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
