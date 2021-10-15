import pandas as pd
import numpy as np
from typing import Union
from PIL import Image
import matplotlib.animation as anim
import matplotlib.pyplot as plt

class DrivingRecord:
    """Driving Record Dataset 
        produces training sets with current imgs and current driving data as inputs and
        next imgs and next driving data as output labels """
    def __init__(self, driving_record : pd.DataFrame) -> None:
        self.driving_record = driving_record
    
    def __len__(self)-> int:
        return len(self.driving_record)-1
    
    def __getitem__(self, idx : Union[int, slice]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = len(self) if idx.stop is None else idx.stop
            step = 1 if idx.step is None else idx.step
            current_imgs = []
            current_data = []
            next_imgs = []
            next_data = []
            
            for i in range(start, stop, step):
                current_img, current_driving_data, next_img, next_driving_data = self[i]
                current_imgs.append(current_img)
                current_data.append(current_driving_data)
                next_imgs.append(next_img)
                next_data.append(next_driving_data)
            
            return np.stack(current_imgs, axis=0), np.stack(current_data, axis=0), np.stack(next_imgs, axis=0), np.stack(next_data, axis=0)

        current_row = self.driving_record.iloc[idx]
        current_img = self._ingest_img(current_row)
        current_driving_data = self._ingest_driving_data(current_row)
        next_row = self.driving_record.iloc[idx+1]
        next_img = self._ingest_img(next_row)
        next_driving_data = self._ingest_driving_data(next_row)
        return current_img, current_driving_data, next_img, next_driving_data

    def get_attribute(self, idx: int, attr: str):
        return self.driving_record.iloc[idx][attr]
    
    def visualize(self):
        pass

    def _ingest_img(self, row : pd.core.series.Series) -> np.ndarray:
        center_img = Image.open(row['center_img'])
        left_img = Image.open(row['left_img'])
        right_img = Image.open(row['right_img'])
        center_img = np.asarray(center_img)
        left_img = np.asarray(left_img)
        right_img = np.asarray(right_img)
        current_img = np.concatenate([left_img, center_img, right_img], axis=-1)
        return current_img/255.0

    def _ingest_driving_data(self, row : pd.core.series.Series) -> np.ndarray:
        return np.asarray(row[['speed', 'steering_angle', 'throttle', 'brake']], dtype=np.float32)