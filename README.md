# AI Project : American sign language letter recognition

## Contributors
- Ulysse MARCANDELLA
- Lucas RIOUX

## Data
You can found the data [here](https://drive.google.com/drive/u/2/folders/1Nna4tQgvKeK0giZZcDhmkxnwpDUfFFbJ).
Put the two data files in a directory named `sign_language_mnist` at the project root.

## Run the project
1. Create a venv at the project root : `python3 -m venv venv`.
2. Install the required dependencies : `pip install -r requirements.txt`.
3. Run the script : `python3 run_model.py -t [d|p|c] -m [integer]`.

`-t` specifies the model type : dummy (d), perceptron (p, default), convolution (c).

`-m` specifies the number of iterations : default is 1. Metrics are averaged across all iterations.
