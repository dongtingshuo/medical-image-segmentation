# Small Batch Overfit Report

- Model: unet
- Device: cpu
- Samples: 8
- Epochs: 50
- Initial loss: 0.720196
- Final loss: 0.344823
- Loss drop: 0.375373
- Initial Dice: 0.393349
- Final Dice: 0.965997
- Dice gain: 0.572648
- Final IoU: 0.935111
- IoU gain: 0.663287
- NaN detected: False
- Prediction samples: /kaggle/working/outputs/sanity_check/overfit_predictions

## Interpretation

- Normal: loss clearly decreases and Dice/IoU increase.
- If loss does not decrease, check mask alignment, learning rate, loss function, and model output logits.
- If predictions remain all black/all white, do not start formal long training.
