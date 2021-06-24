# Food Classification App

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the necessary packages to run food classification.

```bash
pip install flask~=2.0.1
pip install werkzeug~=2.0.1
pip install matplotlib~=3.3.4
pip install numpy~=1.19.5
pip install tensorflow-gpu~=2.5.0
pip install scikit-learn~=0.24.2
```

## Dataset

[Food-11 Image Dataset](https://www.kaggle.com/trolukovich/food11-image-dataset
) from Kaggle was used as the dataset. You can access it from the link below. 


## Usage

You need to create the `all_constants.py` script by referring to the `all_constants.py.sample`.

```python
class AllConstant:
    def __init__(self):
        self.train_set = r""
        self.validation_set = r""
        self.evaluation_set = r""
        self.checkpoints_path = r""
        self.input_shape = (224, 224)
        self.weight_path = r""
        self.lr = 0.0001
        self.batch_size = 32
        self.epochs = 750
        self.train = False
        self.predict = True
        self.evaluation = False
        self.model_type = "VGG19"  # "VGG16"
        self.predict_path = r""
```

You can edit the relevant places for your parameters.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
