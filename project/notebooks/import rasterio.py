import rasterio
import numpy as np

# Открываем GREEN (Band 3)
with rasterio.open("D:/QtCode/qgis_data/b3.tif") as src:
    green = src.read(1).astype(float)
    profile = src.profile  # сохраним метаданные

# Открываем NIR (Band 5)
with rasterio.open("D:/QtCode/qgis_data/b5.tif") as src:
    nir = src.read(1).astype(float)

# NDWI по McFeeters (1996): (Green - NIR) / (Green + NIR)
ndwi = (green - nir) / (green + nir + 1e-6)  # добавлен epsilon для избежания деления на 0

# Маска: NDWI > 0.0 → вода
water_mask = (ndwi > 0.0).astype(np.uint8)

# Обновляем профиль для сохранения бинарной маски
profile.update(dtype=rasterio.uint8, count=1)

# Сохраняем маску
with rasterio.open("water_mask_ndwi.tif", "w", **profile) as dst:
    dst.write(water_mask, 1)

print("COMPLETED")