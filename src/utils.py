from visdom import Visdom
import numpy as np
from PIL import Image


def load_image_numpy(path):
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype='int32')

    # Convert to a (height, width, channel) shape
    # data = data[:, :, None]
    return data


def crop(image):
    '''
    Auxilary function to crop image to non-zero area
    :param image: input image
    :return: cropped image
    '''
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def create_heatmap(dicom_id, fixations, master_sheet, cxr_jpg_path, save_heatmap=None, sigma=150):
    """
    Generate a grayscale heatmap based on fixation points

    :param dicom_id: the DICOM ID for the CXR
    :param fixations: Pandas DF of all fixations
    :param master_sheet: Pandas DF of studies
    :param save_heatmap: If not None, directory to save heatmap to
    :param sigma: Standard deviation for the Gaussian filter. Sigma=150 is used in the original dataset paper
    :return: Grayscale heatmap for given CXR image
    """
    study_info = master_sheet[master_sheet['dicom_id'] == dicom_id]

    # Open the image and show it
    cxr_file_path = study_info['path'].values[0]

    # We're using JPEGs not DICOMS, so change extension
    cxr_file_path = os.path.join(cxr_jpg_path, cxr_file_path[:-4] + '.jpg')

    cxr = load_image_numpy(cxr_file_path)

    # Plot the fixations on the image
    cxr_fixations = fixations[fixations['DICOM_ID'] == dicom_id]
    fixations_mask = np.zeros_like(cxr)

    for i, row in cxr_fixations.iterrows():
        x_coord = row['X_ORIGINAL']
        y_coord = row['Y_ORIGINAL']

        size = 10

        # Make a 10x10 box
        fixations_mask[y_coord - size:y_coord + size, x_coord - size:x_coord + size] = 255

    heatmap = ndimage.gaussian_filter(fixations_mask, sigma)

    if save_heatmap is not None:
        plt.imsave(os.path.join(save_heatmap, '{}-heatmap.jpg'.format(dicom_id)), heatmap)

    # Try scaling the heatmap back to the range [0, 255]
    heatmap = ((heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))) * 255

    return heatmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Visdom Plotting
class VisdomLinePlotter(object):
    def __init__(self, env_name='main', server='localhost', port=8097):
        self.vis = Visdom(server=server, port=port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y, xlabel='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.vis.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')

    def plot_matplotlib(self, plot_name, plt):
        self.plots[plot_name] = self.vis.matplot(plt, env=self.env)

    def plot_text(self, text, title='Text'):
        self.vis.text(text, env=self.env, opts=dict(title=title))


def membership_advantage(confusion_matrix):
    tpr = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
    fpr = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[0][0])

    return tpr - fpr