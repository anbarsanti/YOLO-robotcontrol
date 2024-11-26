import torch
import csv
import numpy as np
import pandas as pd


cls = torch.tensor([ 0.,  0., 47.]) # HBB and OBB have same format for classes

xyxyn = torch.tensor([[0.2672, 0.4354, 0.9411, 0.9989],
        [0.8952, 0.8653, 0.9995, 0.9990],
        [0.2172, 0.7879, 0.2489, 0.8835]]) # 3 classes of HBB format

xyxyxyxyn = torch.tensor([[[0.5924, 0.7221],
         [0.7306, 0.6816],
         [0.6770, 0.3568],
         [0.5388, 0.3973]],

        [[0.7698, 0.6393],
         [0.8842, 0.5910],
         [0.7838, 0.1682],
         [0.6694, 0.2164]],

        [[0.3263, 0.4627],
         [0.3274, 0.6755],
         [0.5153, 0.6737],
         [0.5142, 0.4609]]]) # 3 classes of OBB format

# Just for reference
# for box in r.boxes: # For Horizontal Bounding Boxes
#     cords = box.xyxy[0].tolist()
#     x1, y1, x2, y2 = [round(x) for x in cords]
#     score = box.conf[0].item()  # Assuming the confidence score is available in box.conf
#     cls = r[0].names[box.cls[0].item()]
#     boxes.append([x1, y1, x2, y2, score, cls])
#     scores.append(score)

# print(xyxyn)
# print(xyxyn.shape)
# print(xyxyn[0])
# print(xyxyn[0].tolist())
# print(xyxyn.tolist())
# print(xyxyn.tolist()[0])

# len_cls = len(cls)
# for i in range(len_cls):
#     cxyxyn = [*[(cls[i].tolist())], *(xyxyn[i].tolist())]  # Append class with its HBB



print(xyxyxyxyn)
print(xyxyxyxyn.shape)
xyxyxyxyn_reshape = xyxyxyxyn.reshape(-1, xyxyxyxyn.shape[-1])
print(xyxyxyxyn_reshape)


print("===============")

len_cls = len(cls)
for i in range(len_cls):
    xyxyxyxyn_flatten = (np.array((xyxyxyxyn[i].tolist())).reshape(1,8).tolist())[0] # Flatten the xyxyxyxy
    cxyxyxyxyn = [*[(cls[i].tolist())], *(xyxyxyxyn_flatten)]  # Append class with its HBB
    print(cxyxyxyxyn)
    with open('../file_obb.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([cxyxyxyxyn])
