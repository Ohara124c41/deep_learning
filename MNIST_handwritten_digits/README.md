# MNIST Handwritten Digit Classifier

This repository hosts Udacity's deep learning mini-project for handwritten digit recognition with PyTorch. The notebook walks through loading MNIST, designing and training a multi-layer perceptron, validating the model, and saving the best checkpoint for reuse.

## Project Structure

- `MNIST_Handwritten_Digits-STARTER.ipynb` - primary notebook with all preprocessing, training, evaluation, and rubric commentary.
- `requirements.txt` - minimal dependency pin (ipywidgets) for the provided environment.
- `mnist_mlp_state_dict.pth` - weights/state dict saved after fine-tuning (created once the notebook runs successfully).

## Environment Setup

1. (Optional but recommended) Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Launch Jupyter (Lab or Notebook) in this folder and open `MNIST_Handwritten_Digits-STARTER.ipynb`.

The standard PyTorch and torchvision packages are assumed to be present in the grading workspace; install them locally if needed with `pip install torch torchvision` per your platform's instructions.

## Notebook Workflow

1. **Data loading & exploration** - Applies `ToTensor`, dataset mean/std normalization, and explicit flattening to 784-dimensional vectors. DataLoaders are created for training, validation, testing, plus an auxiliary loader for visualization without normalization.
2. **Modeling** - Implements `MNISTMLP`, a ReLU MLP with 512->256->128 hidden units, BatchNorm, and dropout. Uses negative log likelihood loss with `log_softmax` outputs.
3. **Training** - Trains for 15 epochs using AdamW with weight decay and a ReduceLROnPlateau scheduler, tracking both training and validation metrics.
4. **Evaluation & tuning** - Reports validation curves, evaluates on the hold-out test set, and runs a short fine-tuning stage with a lower learning rate before saving the state dict.

## Results

- **Best validation accuracy:** 98.6%
- **Test accuracy after fine-tuning:** 98.3%
- **Saved model:** `mnist_mlp_state_dict.pth`

These values exceed the rubric requirement of >=90% test accuracy, and the notebook includes textual justifications for preprocessing, observations from training curves, and evaluation notes.

## Re-running / Reusing the Model

- Execute the notebook cells sequentially; any generated artifacts (plots, metrics, saved weights) will appear inline.
- To reuse the trained weights in another script, load them with:
  ```python
  checkpoint = torch.load("mnist_mlp_state_dict.pth", map_location="cpu")
  model = MNISTMLP(**checkpoint["model_config"])
  model.load_state_dict(checkpoint["model_state_dict"])
  ```

Feel free to extend the project with convolutional layers, add validation visualizations, or experiment with alternative optimizers if you want to push accuracy closer to state of the art.
