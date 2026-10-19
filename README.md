# MNIST CNN and Explainable AI

NIT3251 Advanced Data Science group project. The project trains a CNN to recognise
handwritten digits, examines its predictions using saliency maps, Grad-CAM and
LIME, and independently checks its performance on the official MNIST test set.

## Group contributions

| Member | Work |
|---|---|
| Diego Vertiz Padilla | CNN architecture, training, six hyperparameter experiments and model selection |
| Aadil Wazeer | Saliency maps, Grad-CAM, LIME and interpretation of the explanation figures |
| Muhammadh Azwardeen | Preprocessing, restoration of the selected model for evaluation, metrics, error analysis, reference checks and result exports |

## Setup

Use Python 3.11 or 3.12 in a virtual environment, or open the notebooks in Google
Colab. From the repository root, install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The explainability notebook also uses `lime` and `scikit-image`:

```bash
python -m pip install lime scikit-image
```

MNIST is downloaded through `tensorflow.keras.datasets.mnist.load_data()` on its
first use. It contains 60,000 original training images and 10,000 test images.
The training split is stratified into 50,000 training and 10,000 validation
images using `random_state=42`. Inputs are float32 values in the range 0 to 1,
with shape `(N, 28, 28, 1)`. Dataset details: [Keras MNIST documentation](https://keras.io/api/datasets/mnist/).

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/02_cnn_training.ipynb` | Initial architecture and dropout exploration |
| `notebooks/03_cnn_experiments_and_reporting.ipynb` | Six experiments, validation-based model selection and training reference results |
| `notebooks/explainability.ipynb` | Saliency, Grad-CAM and LIME examples |
| `notebooks/04_preprocessing_and_evaluation.ipynb` | Independent Section 3 evaluation with short explanations of each step |

Notebook 03 selects the checkpoint with the lowest **validation loss**. Test
accuracy is not used to choose a model. The selected architecture has two
convolution and pooling blocks, Dense(128), Dropout(0.3) and a ten-class softmax
output, with 421,642 parameters.

The existing explainability notebook expects the checkpoint at
`/content/cnn_selected_for_xai.keras`; upload or copy the selected checkpoint to
that path when running it in Colab. Its plots are in `results/figures/`.

## Run Section 3

Open `notebooks/04_preprocessing_and_evaluation.ipynb` and run its cells in order.
It runs from the repository root or the `notebooks/` directory. In Colab, clone
this repository first and open or run the notebook from the cloned project:

```python
!git clone https://github.com/Vertiz2405/mnist-cnn-explainable-ai.git
%cd /content/mnist-cnn-explainable-ai
```

The same evaluation can be run from the repository root without Jupyter:

```bash
python -m src.section3_evaluation
```

Section 3 restores `results/models/cnn_selected_for_xai.keras` without retraining.
The architecture is recreated before loading its weights to avoid Keras
initializer-configuration differences. It reads the training reference files
under `results/metrics/` and writes separate outputs under `results/section3/`:

- `classification_report.csv`: precision, recall, F1 and support for each digit.
- `confusion_matrix.csv`: actual digits in rows and predictions in columns.
- `ranked_confusions.csv`: all observed mistakes ranked by frequency.
- `evaluation_summary.json`: overall scores, reference checks and checkpoint identifier.
- `confusion_matrix.png`: full test confusion matrix.
- `misclassified_examples.png`: the first 12 mistakes in test order.

For the supplied selected checkpoint, the reference results are:

| Measure | Result |
|---|---:|
| Test images | 10,000 |
| Correct predictions | 9,886 |
| Errors | 114 |
| Accuracy | 98.86% |
| Macro F1 | 0.9886 |

The evaluation raises an error if either accuracy or any confusion-matrix count
does not match the reference. Matching these aggregate results does not establish
that every output probability is identical. Softmax probabilities are model
outputs, not guarantees of correctness.

## Check the evaluation helpers

```bash
python -m unittest discover -s tests -v
```

These checks cover repeated preprocessing, invalid probability arrays, a model
with no mistakes, and a changed confusion matrix hidden behind matching accuracy.
Running Section 3 also checks the actual checkpoint against all 10,000 test images.

## Scope and remaining work

The current explanation notebook implements three XAI methods. It does not
perform adversarial attacks or quantitative robustness evaluation. Those results
must not be claimed without running the relevant experiments. The first 12
misclassified images are examples, not a representative sample of every error.
Future model choices should use validation data rather than repeated test tuning.

The group report is stored in `reports/`. Section 3 methods and interpretation are
also recorded in `reports/preprocessing_evaluation.md`.
