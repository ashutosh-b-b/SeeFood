# Setup Instructions: 
1. `cd SeeFoodServer/`
2. Make `setup.sh` executable by `chmod +x setup.sh` 
3. Make a directory `model_artifacts` where you store the following:
    - `model_state.pth` : The state of the model that is trained using `SeeFoodModel` 
    - `idx_to_label.json` : Access the idx to label by doing `train_dataset.cat_id_to_label` 
    - `model_config.json`: The model configuration that will eventually be passed to `configure_model` from SeeFoodModel.
4. Run `uvicorn app.main:app --reload`
5. To test the endpoint run `python test_endpoint.py <your image path>`
6. See `localhost:8080/docs` for API information.