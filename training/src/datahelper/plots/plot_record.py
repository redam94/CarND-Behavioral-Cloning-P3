from IPython.core.display import Video
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from ..datasets import DrivingRecord

def plot_record(current_frame:np.ndarray, current_input:np.ndarray, next_frame:np.ndarray, next_input:np.ndarray) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    plt.tight_layout()
    ax[0][0].imshow(current_frame[:, :, :3])
    ax[0][1].imshow(current_frame[:, :, 3:6])
    ax[0][2].imshow(current_frame[:, :, 6:])
    ax[0][0].set_title('Left Current Img')
    ax[0][1].set_title('Center Current Img')
    ax[0][2].set_title('Right Current Img')
    ax[1][0].imshow(next_frame[:, :, :3])
    ax[1][1].imshow(next_frame[:, :, 3:6])
    ax[1][2].imshow(next_frame[:, :, 6:])
    ax[1][0].set_title('Left Next Img')
    ax[1][1].set_title('Center Next Img')
    ax[1][2].set_title('Right Next Img')
    [ax.get_yaxis().set_visible(False) or ax.get_xaxis().set_visible(False) for ax in fig.axes]
    plt.show()    

def animate_record(record: DrivingRecord, name: str = 'test_anim.mp4') -> None:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    first_left_frame = record[0][0][:, :, :3]
    first_center_frame = record[0][0][:, :, 3:6]
    first_right_frame = record[0][0][:, :, 6:]
    left_plot = ax[0].imshow(first_left_frame)
    center_plot = ax[1].imshow(first_center_frame)
    right_plot = ax[2].imshow(first_right_frame)
    record_length = len(record)
    
    def _animate_func(i):
        
        if i % 30 == 0:
            print( '.', end ='' )
        ith_image = record[i][0]
        left_plot.set_array(ith_image[:, :, :3])
        center_plot.set_array(ith_image[:, :, 3:6])
        right_plot.set_array(ith_image[:, :, 6:])
    
        return [left_plot, center_plot, right_plot]
    
    video = anim.FuncAnimation(
                               fig, 
                               _animate_func, 
                               frames = record_length,
                               interval =70, # in ms
                               )

    video.save(name, fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()
    return Video(name)
