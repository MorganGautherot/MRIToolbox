import os
import numpy as np
import nibabel as nib
import pandas as pd
from operator import itemgetter
from operator import add

def get_pos_voxels(mask):
    """
    Compute x,y,z coordinates of a binary mask
    Input:
       - mask: binary mask

    Output:
       - list of tuples containing the (x,y,z) coordinate for each of the input voxels
    """

    indices = np.stack(np.nonzero(mask), axis=1)
    indices = np.array([tuple(idx) for idx in indices])

    return indices

def load_masks(labels_path, balance=True):
    """
    load lesion and non lesion masks, output also lesion centers for voxels.
    Inputs:
        - df_subject_images: dataFrame ['Id', 'Modalities', 'Path'], MRI path for every Id
        - selected_voxels: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
        - scans: list of every id in the DataFrame
        - balance: boolean, if you want a balance dataset

    Output:
        - lesion_masks: boolean matrix identify lesion voxels [n, x, y ,z] (n: number of examples; (x,y,z): coordinates)
        - lesion_centers: Matrix containing the center coordonate for every lesion patches [n, x, y, z] (n: number of patches, (x, y, z): coordonate)
        - no_lesion_centers: Matrix containing the center coordonate for every no lesion patches [n, x, y, z] (n: number of patches, (x, y, z): coordonate)
    """
    lesion_masks = np.array([nib.load(lbls).get_data().astype(dtype=np.bool) for lbls in  labels_path])

    lesion_centers = np.array([get_pos_voxels(msks) for msks in lesion_masks])
    
    if balance : 
        min_center = np.min([cntrs.shape[0] for cntrs in lesion_centers])

        for i in range(lesion_centers.shape[0]):
            indices = np.random.permutation(range(0, lesion_centers[i].shape[0])).tolist()[:min_center]
            lesion_centers[i] = np.array(itemgetter(*indices)(lesion_centers[i]))
            
    nb_label_patches = np.sum([cntrs.shape[0] for cntrs in lesion_centers])

    return lesion_masks, np.concatenate(lesion_centers,axis=0), nb_label_patches


def select_voxels(images_paths, nb_label_patches, threshold=0.5):
    """
    Select voxels based on a intensity threshold
    Inputs:
        - subject_images_paths: list containing all subject image paths for a single modality
        - threshold: minimum threshold to apply by default 0.5

    Output:
        - selected_voxels: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
    """

    # load images
    images = np.array([nib.load(img).get_data() for img in images_paths])

    # select voxels with intensity higher than threshold
    selected_voxels = np.array([np.where(img > threshold, 1, 0) for img in images])

    return images, selected_voxels

def load_nifti(images_path, lesion_masks, nb_label_patches, threshold_selection_voxel):
    
    matrix, selected_voxels = select_voxels(images_path, nb_label_patches, threshold_selection_voxel) 
    
    mask = np.where(np.sum(lesion_masks, axis=0)>=1, 1, 0)
       
    no_lesion_masks = np.array([np.logical_and(np.logical_not(mask), selected) for selected in selected_voxels])
    
    no_lesion_centers = [get_pos_voxels(mask) for mask in no_lesion_masks]
    
    for i in range(np.array(no_lesion_centers).shape[0]):
        indices = np.random.permutation(range(0, no_lesion_centers[i].shape[0])).tolist()[:nb_label_patches//np.array(no_lesion_centers).shape[0]]
        no_lesion_centers[i] = np.array(itemgetter(*indices)(no_lesion_centers[i]))
    
    return matrix, np.concatenate(no_lesion_centers,axis=0)

def get_patches(image, centers, patch_size):
    """
    Get image patches of configurable size based on a set of centers
    Input:
       - image: matrice of the intensity of the images [x, y, z]
       - centers: coordonates of the patch's center
       - patch_size: size of the patch [p, p, p] (default value [15, 15, 15])
    Output:
       - patches: matrice with all the patches [n, p, p, p] (n is the number of patches)
    """
    # If the size has even numbers, the patch will be centered. If not, it will try to create an square almost centered.
    # By doing this we allow pooling when using encoders/unets.
    patches = []
    
    if [len(center) == len(patch_size) for center in centers]:
        patch_half = tuple([idx//2 for idx in patch_size])
        new_centers = [map(add, center, patch_half) for center in centers]
        padding = tuple((idx, size-idx) for idx, size in zip(patch_half, patch_size))
        new_image = np.pad(image, padding, mode='constant', constant_values=0)
        slices = [[slice(c_idx-p_idx, c_idx+(s_idx-p_idx)) for (c_idx, p_idx, s_idx) in zip(center, patch_half, patch_size)] for center in new_centers]
        patches = np.array([new_image[tuple(idx)] for idx in slices])

    return patches

def generate_patches(matrix, lesion_centers, no_lesion_centers, lesion_masks, patch_size):
    
  
    y_neg_patches = np.stack([np.array(get_patches(image, no_lesion_centers, patch_size)) for image in lesion_masks], axis=4)

    y_pos_patches = np.stack([np.array(get_patches(image, lesion_centers, patch_size)) for image in lesion_masks], axis=4)

    y = np.concatenate([y_neg_patches, y_pos_patches] , axis=0)

    # load our undersample negative dataset (no lesion voxel s)
    x_neg_patches = np.stack([np.array(get_patches(image, no_lesion_centers, patch_size)) for image in matrix], axis=4)
    
    # load all positive samples (lesion voxels)
    x_pos_patches = np.stack([np.array(get_patches(image, lesion_centers, patch_size)) for image in matrix], axis=4)

    # concatenate positive and negative patches for each subject
    x = np.concatenate([x_neg_patches, x_pos_patches] , axis=0)
    
    return x, y

def save_patches(x, y, save_path, dataframe):
    
    path = save_path+'data/'+str(labels[0])

    os.system('mkdir '+path)
    os.system('mkdir '+path+'/x/')
    os.system('mkdir '+path+'/y/')
    
    cmpt = 0
    for img_x, img_y in zip(x, y):
        
        path_img_x = path+'/x/'+str(cmpt)+'.npy'
        np.save(path_img_x, img_x)

        path_img_y = path+'/y/'+str(cmpt)+'.npy'
        np.save(path_img_y, img_y)
        
        dataframe = dataframe.append(pd.DataFrame([[images['id'],
                                                    cmpt, 
                                                    images.shape[0]-1,
                                                    labels.shape[0]-1,
                                                    path_img_x,
                                                    path_img_y]],
                                                    columns = ('participant_id', 
                                                               'image_id', 
                                                               'x_modalities', 
                                                               'y_modalities', 
                                                               'x_path', 
                                                               'y_path')))
        cmpt+=1
    return dataframe


def create_patches(images_path, labels_path, save_path, threshold_selection_voxel=0.5, patch_size=(11, 11, 11), balance=True):

    dataframe = pd.DataFrame([], columns=('participant_id', 'image_id', 'x_modalities', 'y_modalities', 'x_path', 'y_path'))
    os.system('mkdir '+save_path+'/data/')
    for images, labels in zip(images_path.iterrows(), labels_path.iterrows()):
        images = images[1]
        labels = labels[1]
        output_modalities = np.squeeze(labels.shape)-1

        lesion_masks, lesion_centers, nb_label_patches = load_masks(labels[1:], balance)

        input_modalities = np.squeeze(images.shape)-1

        matrix, no_lesion_centers = load_nifti(images[1:], lesion_masks, nb_label_patches, threshold_selection_voxel) 

        x, y = generate_patches(matrix, lesion_centers, no_lesion_centers, lesion_masks, patch_size)

        dataframe = save_patches(x, y, save_path, dataframe)

    dataframe.to_csv(save_path+'/dataframe.csv')
