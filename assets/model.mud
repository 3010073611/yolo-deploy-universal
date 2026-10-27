[basic]
type = axmodel
model_npu = model_npu.axmodel
model_vnpu = model_vnpu.axmodel

[extra]
model_type = yolo11
type = detector
input_type = rgb
labels = class0, class1, class2

input_cache = true
output_cache = true
input_cache_flush = false
output_cache_inval = true

mean = 0,0,0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
