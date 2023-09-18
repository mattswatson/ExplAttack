import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import pydicom

import os

from utils import create_heatmap, load_image_numpy

class GazeXrayDataset(Dataset):
    def __init__(self, gaze_path, cxr_jpg_path, cxr_transforms=None, heatmap_transforms=None, heatmap_threshold=None,
                 sigma=150, generated_heatmaps_path=None, return_binary_heatmap=False, repeat=None):
        """
        :param gaze_path: Path to base EGD dataset
        :param cxr_jpg_path: Path to the original CXR images in JPEG format
        :param cxr_transforms: A list of torchvision.transforms to perform on the CXRs
        :param heatmap_threshold: Only include heatmap pixels larger than this threshold
        :param sigma: Integer value to use for standard deviation of the Gaussian used when generating heatmaps
        :param generated_heatmaps_path: Path to pre-generated heatmaps. If None, new heatmaps are generated from raw
        fixations
        :param repeat: (int), number of time to repeat the dataset
        """
        super(GazeXrayDataset, self).__init__()

        self.gaze_path = gaze_path
        self.cxr_jpg_path = cxr_jpg_path
        self.cxr_transforms = cxr_transforms
        self.heatmap_transforms = heatmap_transforms
        self.heatmap_threshold = heatmap_threshold
        self.sigma = sigma
        self.generated_heatmaps_path = generated_heatmaps_path
        self.return_binary_heatmap = return_binary_heatmap
        self.repeat = repeat

        # These are the three labels we're interested in
        self.class_names = ['Normal', 'CHF', 'pneumonia']

        # CSV file that contains case information
        self.master_sheet = pd.read_csv(os.path.join(self.gaze_path, 'master_sheet.csv'))

        # CSV file with fixations - we only need this if we're not using generated heatmaps
        if not self.generated_heatmaps_path:
            self.fixations = pd.read_csv(os.path.join(self.gaze_path, 'fixations.csv'))
        else:
            # If we're using generated heatmaps, use only CXRs we also have the heatmaps for
            dicoms_with_heatmaps = [p for p in os.listdir(self.generated_heatmaps_path) if
                                    os.path.isdir(os.path.join(self.generated_heatmaps_path, p))]

            self.master_sheet = self.master_sheet[self.master_sheet['dicom_id'].isin(dicoms_with_heatmaps)]

        self.dicom_ids = np.array(self.master_sheet['dicom_id'].unique())

        if self.repeat:
            self.dicom_ids *= self.repeat

    def __len__(self):
        return len(self.dicom_ids)

    def get_heatmap(self, dicom_id, return_binary_heatmap=False):
        """
        Returns either a pre-generated or newly-generated heatmap for the given DICOM ID, depending on class
        variables
        :param dicom_id: ID of the CXR image to generate the heatmap for
        :return: 1D numpy array
        """

        # If we're given a generated_heatmaps_path then we just want to load the generated heatmap
        if self.generated_heatmaps_path:
            heatmap_path = os.path.join(self.generated_heatmaps_path, dicom_id, 'heatmap.png')

            try:
                heatmap = load_image_numpy(heatmap_path)
            except FileNotFoundError:
                print('Heatmap for CXR {} not found at {}! Heatmap will be all zeros.'.format(dicom_id, heatmap_path))

            # We don't care about the alpha channel
            heatmap = heatmap[:, :, :3]
        else:
            # Otherwise we need to generate it from scratch
            heatmap = create_heatmap(dicom_id, self.fixations, self.master_sheet, self.cxr_jpg_path, sigma=self.sigma)

        # Torch expects images to be of shape (C, H, W)
        if len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
            heatmap = np.transpose(heatmap, (2, 0, 1))

        if len(heatmap.shape) != 3:
            heatmap = np.stack([heatmap, heatmap, heatmap], axis=0)

        # If we're returning a binary heatmap, we should just return the fixation locations as a binary image
        if return_binary_heatmap:
            fixations_mask = np.zeros_like(heatmap)[:, :, 0]

            cxr_fixations = self.fixations[self.fixations['DICOM_ID'] == dicom_id]

            for i, row in cxr_fixations.iterrows():
                x_coord = row['X_ORIGINAL']
                y_coord = row['Y_ORIGINAL']

                size = 10

                # Make a 10x10 box
                fixations_mask[y_coord - size:y_coord + size, x_coord - size:x_coord + size] = 255

            return heatmap, fixations_mask

        return heatmap

    def __getitem__(self, item):
        """
        :param item: ID (not DICOM ID!) to return
        :return cxr: numpy array of the CXR image
        :return heatmap: numpy array of the heatmap
        :return labels: numpy array of a one-hot encoding of the labels ['Normal', 'CHF', 'pneumonia']
        """
        # This is the DICOM ID we want to get
        dicom_id = self.dicom_ids[item]

        # Let's get a one-hot encoded label: [normal, chf, pneumothorax]
        study_info = self.master_sheet[self.master_sheet['dicom_id'] == dicom_id]
        labels = np.array(study_info[self.class_names], dtype='int32')

        cxr_file_path = study_info['path'].values[0]
        cxr_file_path = os.path.join(self.cxr_jpg_path, cxr_file_path[:-4] + '.jpg')
        cxr = load_image_numpy(cxr_file_path).astype('float32')

        if self.return_binary_heatmap:
            heatmap, binary_heatmap = self.get_heatmap(dicom_id, self.return_binary_heatmap)
            heatmap = heatmap.astype('float32')
        else:
            heatmap = self.get_heatmap(dicom_id).astype('float32')

        # If we've loaded in a pre-generated heatmap, our CXR will need resizing
        if cxr.shape != heatmap.shape[1:]:
            if not self.generated_heatmaps_path:
                # If we haven't loaded in a pre-generated heatmap but we still have the wrong shape, something has
                # gone wrong
                raise Exception('CXR had shape {} and heatmap had shape {}: if generating heatmaps, these should be'
                                'the same size!'.format(cxr.shape, heatmap.shape))

            # Otherwise all we need to to is resize the CXR (last shape element is channel dimension)
            cxr = cv2.resize(cxr, (heatmap.shape[2], heatmap.shape[1]))

        # We probably want colour images
        if len(cxr.shape) != 3:
            cxr = np.stack([cxr, cxr, cxr], axis=0)

        cxr = torch.tensor(cxr)
        heatmap = torch.tensor(heatmap)

        # We want images in the range [0, 1] not [0, 255]
        if torch.max(cxr) > 1:
            cxr = cxr / 255

        if torch.max(heatmap) > 1:
            heatmap = heatmap / 255

        if self.cxr_transforms:
            cxr_transform = transforms.Compose(self.cxr_transforms)
            cxr = cxr_transform(cxr)

        if self.heatmap_transforms:
            heatmap_transform = transforms.Compose(self.heatmap_transforms)
            heatmap = heatmap_transform(heatmap)

        if self.return_binary_heatmap:
            return cxr, heatmap, binary_heatmap, labels

        return cxr, heatmap, labels


class XrayDataset(Dataset):
    def __init__(self, data_path, cxr_transforms=None, return_blank_heatmap=False):
        super(XrayDataset, self).__init__()

        self.data_path = data_path
        self.cxr_transforms = cxr_transforms
        self.return_blank_heatmap = return_blank_heatmap

        # These are the three labels we're interested in
        self.class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Pneumonia']

        label_df = pd.read_csv(os.path.join(self.data_path, 'chexpert-labels.csv'))

        # If we have nothing in the No Finding column, assume we found something
        label_df['No Finding'] = label_df['No Finding'].fillna(0)

        # Remove rows where there are no annotations for any of our labels
        label_df = label_df[label_df[self.class_names].notnull().all(1)]

        # Fill NANs to 0
        label_df = label_df.fillna(0)

        # -1 means uncertain, change to 0
        label_df = label_df.replace(-1, 0)

        # Set subject and study ids to strings
        label_df['subject_id'] = label_df['subject_id'].astype(str)
        label_df['study_id'] = label_df['study_id'].astype(str)

        # Get a list of DICOM IDs
        all_rows = []
        for i, row in label_df.iterrows():
            labels = row[self.class_names]
            dicom_paths = self._get_dicom_ids(row)

            if dicom_paths is not None:
                rows = []
                for p in dicom_paths:
                    r = [p] + list(labels)
                    rows.append(r)

                all_rows += rows

        self.df = pd.DataFrame(all_rows, columns=['path'] + self.class_names)
        print(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.loc[item]

        labels = row[self.class_names]

        cxr = load_image_numpy(row['path']).astype('float32')

        # We probably want colour images
        if len(cxr.shape) != 3:
            cxr = np.stack([cxr, cxr, cxr], axis=0)

        cxr = torch.tensor(cxr)

        # We want images in the range [0, 1] not [0, 255]
        if torch.max(cxr) > 1:
            cxr = cxr / 255

        if self.cxr_transforms:
            cxr_transform = transforms.Compose(self.cxr_transforms)
            cxr = cxr_transform(cxr)

        if self.return_blank_heatmap:
            return cxr, torch.zeros_like(cxr), torch.tensor(labels)[None, :]

        return cxr, torch.tensor(labels)[None, :]

    def _get_dicom_ids(self, row):
        subject_id_prefix = 'p' + row['subject_id'][:2]
        subject_id = 'p' + row['subject_id']
        study_id = 's' + row['study_id']
        path = os.path.join(self.data_path, 'files', subject_id_prefix, subject_id, study_id)

        try:
            files = os.listdir(path)
        except FileNotFoundError:
            print(row)
            path = os.path.join(self.data_path, 'files', subject_id_prefix[:-2], subject_id[:-2], study_id[:-2])
            files = os.listdir(path)

        all_paths = [os.path.join(path, f) for f in files if str(f).endswith('.jpg')]

        return all_paths


class XrayDCMDataset(Dataset):
    def __init__(self, data_path, collated_labels_path, label, mention, normalise=True, include_filenames=False,
                 only=None, use_not_mentioned=True):
        super(XrayDCMDataset, self).__init__()

        self.normalise = normalise
        self.include_filenames = include_filenames
        self.only = only

        self.collated_labels = pd.read_csv(collated_labels_path)
        self.data_path = data_path

        self.img_paths = self.__get_images_with_label(label, mention)

        # We need to get the others too, and then label them correctly
        # Assume if not mentioned, it isn't present
        # Label positive images first
        rows_paths = []
        rows_labels = []

        if self.only is None or self.only == mention:
            print('only: {}, mention: {}'.format(self.only, mention))
            for path in self.img_paths:
                rows_paths.append([path])
                rows_labels.append([1])

        print('normal rows:', len(rows_labels))
        # Now get the rest of the paths
        if use_not_mentioned:
            other_rows = self.collated_labels[self.collated_labels[label] != mention]
        else:
            other_mention = 1.0 - mention
            other_rows = self.collated_labels[self.collated_labels[label] == other_mention]
            print('other_rows:', len(other_rows))

        other_paths = other_rows['dcm_path']

        if self.only is None or self.only != mention:
            print('only: {}, mention: {}'.format(self.only, mention))
            for path in other_paths:
                rows_paths.append([path])
                rows_labels.append([0])

        # Create the dataframes
        self.df = pd.DataFrame(rows_paths, columns=['path'])
        self.labels = pd.DataFrame(rows_labels, columns=['label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # We need to return the actual images, not the paths
        if type(idx) is list:
            images = []

            for i in idx:
                row = self.df.iloc[i]
                path = os.path.join(self.data_path, row['path'])
                dcm = pydicom.dcmread(path)

                img = dcm.pixel_array.astype(np.float16)
                img = torch.Tensor(img)
                images.append(img)

            return images, self.labels[idx]

        row = self.df.iloc[idx]
        path = os.path.join(self.data_path, row['path'])

        filename = os.path.basename(path)

        dcm = pydicom.dcmread(path)

        # For now, just keep the pixel data
        img = dcm.pixel_array
        image_2d = img.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        # plt.imshow(image_2d_scaled)
        # plt.show()

        img = Image.fromarray(image_2d_scaled).convert('F')

        # Densenet takes in RGB images
        img = img.convert("RGB")

        pixels = list(img.getdata())
        width, height = img.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

        # The x-rays are all different dimensions
        # So make them the same size here
        # Normalise to range expected by densenet
        if self.normalise:
            composed = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Foolbox wants the unnormalised images
            composed = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])

        img = composed(img)

        if self.include_filenames:
            return img, self.labels.iloc[idx].item(), filename
        else:
            return img, self.labels.iloc[idx].item()

    # Get all images with desired label
    def __get_images_with_label(self, label, mention):
        # label = type of diagnosis
        # mention = 1, 0, -1, None (=NaN)

        labels = list(self.collated_labels.columns)[2:]

        if label not in labels:
            raise Exception('label should be a proper label generated by CheXpert')

        if mention not in [1, 0, -1, None]:
            raise Exception('mention should be 1, 0, -1 or None')

        rows = self.collated_labels[self.collated_labels[label] == mention]
        imgs = rows['dcm_path']

        imgs_list = []
        for img in imgs:
            imgs_list.append(img)

        return imgs_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random

    dataset = GazeXrayDataset('/media/hdd/mimic-cxr-eye', '/media/hdd/mimic-cxr-jpg',
                              cxr_transforms=[transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])],
                              generated_heatmaps_path=None)

    #for i in range(len(dataset)):
    #    cxr, heatmap, labels = dataset[i]
    #    print(i, 'done')

    idx = random.randint(0, len(dataset))

    plt.imshow(dataset[idx][0].permute(1, 2, 0))
    plt.savefig('./test-cxr.png', bbox_inches='tight')

    print(dataset[idx][1].shape)
    plt.imshow(dataset[idx][1].permute(1, 2, 0))
    plt.savefig('./test-heatmap.png', bbox_inches='tight')
