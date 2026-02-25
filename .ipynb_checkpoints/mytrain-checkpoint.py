from ultralytics import YOLO
import os
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9' 
# if __name__ == '__main__':

#     model = YOLO('yolo12-vanillanet-AWAU.yaml')
#     model.train(
#         data = r'renshen.yaml',
#         epochs=800,
#         imgsz=640,
#         batch=-1,
#         amp=False,
#         project='results',
#         name='yolov12n-vanillanet-AWAU',
     
#     )

    
# model = YOLO(r'E:\deeplearning\yolo\ultralytics-8.3.163\runs\detect\train5\weights\best.pt')
# print(model.names)
model_names = ['yolo12-vanillanet-SPPF_MixPool']
for model_name in model_names:
    model = YOLO(model_name+'.yaml')
    model.train(
        data=r'renshen.yaml',
        epochs=800,
        imgsz=640,
        batch=64,
        project='results',
        name=model_name    
)