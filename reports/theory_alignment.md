# Theory and implementation check

## Lecture concepts used

| Concept | Course material | Project implementation |
|---|---|---|
| CNN image features | Session 2, slides 4 and 6 to 16 | Two 3 x 3 convolution layers use ReLU and same padding; 2 x 2 max pooling reduces 28 x 28 inputs to 7 x 7 feature maps. |
| Multiclass outputs | Session 2, slides 3 and 18 to 20 | Ten softmax outputs represent digits 0 to 9. Predictions use the largest output. |
| Pixel scaling | Session 2, slide 25 | Raw uint8 pixels are divided by 255 and stored as float32 values in the range 0 to 1. |
| Hyperparameters | Session 2, slide 26 | Notebook 03 compares dense-layer size, dropout and learning rate while retaining the same convolutional backbone. |
| Overfitting controls | Session 4, slides 55 to 63 | Experiments compare smaller dense layers and Dropout(0.3), with early stopping based on validation loss. |
| Independent evaluation | Project development and optimisation instructions | Section 3 uses the selected checkpoint on the official test set, without retraining or selecting a model from test results. |

The lecture example uses one-hot labels with categorical cross-entropy. This
project keeps integer digit labels, so sparse categorical cross-entropy is the
appropriate equivalent. Integer labels do not need one-hot conversion when this
loss is used. See the [Keras loss documentation](https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class).

## Evaluation checks

The independent Section 3 implementation reproduces 98.86% accuracy, 114 errors
and the complete 10 x 10 reference confusion matrix. Accuracy, precision, recall
and F1 measure classification performance; they do not establish the quality of
an explanation or resistance to adversarial inputs. Dividing pixels by 255 is
scaling, not data augmentation. Loading a trained checkpoint for inference is
not another training experiment.

## Explainability review

The existing notebook computes absolute input gradients for saliency, weights
the second convolution layer's feature maps using pooled gradients for Grad-CAM,
and uses SLIC regions and 1,000 perturbed samples for LIME. These are separate
explanation methods. Their colourful regions are not proof of causal reasoning.

The Grad-CAM implementation differentiates the softmax probability. Saturated
probabilities can produce weak gradients. A near-empty heatmap must not be
interpreted as proof that the model ignored the image. LIME requests **up to**
ten positive regions; it need not select exactly ten. The ten digit examples are
the first test occurrence of each label, rather than a filter that explicitly
searches for correct predictions.

Primary sources for the methods:

- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- [Why Should I Trust You?: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)

## Report checks still needed

- Adversarial analysis is listed among the chosen topic's deliverables. The
  current code and report do not include completed adversarial testing. It must
  remain recorded as unfinished until experiments are run.
- Add bibliography entries for the Grad-CAM and LIME citations numbered [4] and
  [5] in the group report.
- Digit 7 precision is `0.9922` in the numerical CSV. Use that value when
  transcribing or checking the report's tables.
- Check the final report against the shared 2,000-word limit and 15-page limit.
  Adding a teammate's section changes both totals, so an earlier count is not a
  final submission check.

This review records the current implementation. It does not claim that every
course topic must be implemented in the MNIST project.
