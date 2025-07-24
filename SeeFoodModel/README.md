# SeeFood : Single Shot Detection Model

## Dataset: 
We use IndianFoodNet-30 dataset from roboflow.
Reference:
Roboflow Universe - IndianFoodNet-30 Dataset 2023
Agarwal, Ritu and Bansal, Nikunj and Choudhury, Tanupriya and Sarkar, Tanmay and J.Ahuja, Neelu


## Model: 
We use a tiny version of Single Shot Detection model.

### Run Instructions: 
- Unzip the dataset inside `SeeFoodModel`
    - `unzip IndianFoodNet.zip SeeFoodDataset`
- Start MLFlow tracking server:
    - `mlflow server --host 127.0.0.1 --port 8081`
- Setup `params.json` for configuration of a run. If you have multiple GPU devices, configure different device inside `config.device`
- Run `python train_model.py --id 0` 
