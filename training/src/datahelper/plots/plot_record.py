import matplotlib.pyplot as plt
import numpy as np

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
