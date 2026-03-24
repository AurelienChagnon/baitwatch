from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(data="raw_data/all_data/data.yaml", epochs=50, batch=16, imgsz=(384, 640), patience=5)
