from ultralytics import YOLO


if __name__ == '__main__':

    model_names = ['yolo12-vanillanet-SPPF_MixPool']
    for model_name in model_names:
        model = YOLO(model_name+'.yaml')
        model.train(
            data=r'renshen.yaml',
            epochs=800,
            imgsz=640,
            batch=-1,
            project='results',
            name=model_name
    )