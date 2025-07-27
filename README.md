# SeeFood 

<p align = "center">
<img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExanhvMmV5bmc5anJtd3djMWVteXh6am0wczZmYWcxaXVtcTV2NjJ5OCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0Iy9iqThC2ueLTkA/giphy.gif" alt="Jin Yang" width="300"/>
</p>

Inspired by Jin Yang's app from Silicon Valley, SeeFood is a food classifier trained on Indian food items. 
Its an end to end pipeline with:

- ### SeeFoodModel: 
    - Single Shot Detection with anchor boxes model definitions
    - Model iterations and experiment tracking. 
    - Dataset object for `IndianFoodNet30`
    - Stack: PyTorch, MLFlow

- ### SeeFoodServer: 
    - FastAPI server that runs the PyTorch model.
    - Runs prediction with `base64` encoded image in POST request.
    - Stack: FastAPI, pydantic

- ### SeeFoodModelApp:
    - The mobile app frontent synced with server.
    - Stack: Expo 

## Working Demo
<p align = "center">
<img src="assets/seefood.gif" alt="App Demo" width="180" height = "320" />
</p>
