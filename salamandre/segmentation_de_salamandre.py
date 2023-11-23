
#Memoire de Sarra Laksaci
"""
Les modifications ne seront pas enregistrées
READ ME:
Ce notebook permet d'utiliser le modèle de segmentation de salamandres sur des images d'observations. Afin d'extraire le corps des salamandres présentes sur les images, il suffit d'importer les images dans le dossier "images" du drive avant d'exécuter la totalité des cellules de code ci-dessous.

Les images admises par le modèle doivent inclure une seule salamandre, orientée de façon à avoir la tête vers le haut.

Avant de lancer le code, on peut voir plus bas un exemple de résultats produits par le modèle IA. Sur ces visualisations, on a à gauche l'image récupérée du dossier et à droite le corps de la salamandre extrait. Ces images de droite sont sauvegardées et disponibles au téléchargement dans le fichier zip de nom "segmentation_salamandres.zip".

MODIFICATION: La cellule de code numéro 3 débutant par le commentaire "-MODIFICATION-" reprend les liens d'accès aux images et au modèle. En cas d'erreur "no such file or directory", vérifiez que les liens repris dans cette cellule sont bien en accord avec la localisation des fichiers dans le drive. Le dossier partagé "Segmentation_de_salamandre" est automatiquement répertorié dans les dossiers "partagés avec moi" du drive. Assurez-vous qu'il est disponible dans "Mon Drive". Pour ce faire, il suffit de créer un raccourci de "Segmentation_de_salamandre" dans "Mon Drive".

EXECUTION: Toutes les cellules doivent être exécutées afin de récupérer les images souhaitées.
    Pour ce faire, vous pouvez soit (1) les exécuter une par une dans le bon ordre en cliquant sur le bouton "run" qui apparaît lorsque vous placez le curseur sur le coin haut-gauche de la cellule marquée par [ ], soit (2) tout exécuter en cliquant sur  "Runtime"  ->  "Runall"  dans la barre de menus.
TELECHARGEMENT: Afin de récupérer les images produites par le modèle IA, il suffit de télécharger le fichier zip précédemment mentionné.
    Celui-ci peut être téléchargé à partir de l'onglet "Fichiers" à gauche de l'interface en cliquant sur l'icône de dossier.
    Une fois dans l'onglet "Fichiers", placez le curseur sur le fichier zip et cliquez sur les trois points qui apparaissent, puis cliquez sur "Télécharger".
"""

#import dependencies

import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from pillow_heif import register_heif_opener

from keras.models import load_model
from skimage.measure import label, regionprops, regionprops_table

#-MODIFICATION-
root_path="../images/"
model_path="u_net_upright_data-2.h5"

images =[root_path+ doc for doc in os.listdir(root_path)]
register_heif_opener()

# Define custom loss function
def dice_coef(y_true, y_pred):
    """ Dice's coefficient
    Args:
        y_true: correct mask of the salamander
        y_pred: predicted mask of the salamander
    """
    smooth = 10.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """ Dice's loss
    Args:
        y_true: correct mask of the salamander
        y_pred: predicted mask of the salamander
    """
    return 1 - dice_coef(y_true, y_pred)

# Load the saved model and specify the custom loss function
with tf.keras.utils.custom_object_scope({'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}):
    loaded_model = load_model( model_path)

def non_stretching_resize(img,cmap,desired_size=256):

    old_size= img.size
    ratio= float(desired_size)/max(old_size)
    new_size =tuple([int(x * ratio) for x in old_size])

    im = img.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new(cmap, (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))

    img_array=np.asarray(new_im)/255.
    return new_im
def remove_outliers(mask, min_size=100):
    mask = (mask * 255).astype(np.uint8)

    # Find connected components in the mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Filter out small regions
    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered_mask[labels == label] = 255

    return filtered_mask
def dilation(msk):
    msk = (msk * 255).astype(np.uint8)

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(msk, disk)

    return img



def make_dataset():
  x = []
  for i,image in enumerate(images):
    print("\r"+str(i)+"/"+str(len(images)),end="")
    if image==".DS_Store":
      print('found the impostor')
      pass
    image = Image.open(os.path.join(root_path,image))#.convert('L')
    image = np.asarray(non_stretching_resize(image,"RGB"))/255.
    x.append(image)
  return np.array(x)
data= make_dataset()

def fitted_line(msk_sample,plot=True,scatter=False,ax_rotation=False):
  """
      computed fitted line that goes through salamander body mask
      msk: the binary mask of the salamander body
      plot: boolean, true if we want to splot the result
      return: the coordinates of line start and line end
  """
  contours, hierarchy = cv2.findContours(msk_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  rows, cols = msk_sample.shape

  # Calculate endpoints of the line
  vx, vy, x, y = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)

  starty = 0

  if vy!=0:
    startx = int((starty - y) * (vx / vy) + x)
    bottomy = rows - 1
    bottomx = int((bottomy - y) * (vx / vy) + x)
  else:
    startx= bottomx =x
    bottomy = rows - 1

  # Check if the line goes out of bounds
  if startx < 0:
      startx, starty = 0, int(y - (x / vx) * vy)
  elif startx >= cols:
      startx, starty = cols - 1, int(y + ((cols - 1 - x) / vx) * vy)
  if bottomx < 0:
      bottomx, bottomy = 0, int(y - (x / vx) * vy)
  elif bottomx >= cols:
      bottomx, bottomy = cols - 1, int(y + ((cols - 1 - x) / vx) * vy)
  if plot:
  # Plot the line
    fig, ax = plt.subplots()
    ax.imshow(msk_sample, cmap='gray')
    ax.plot([startx, bottomx], [starty, bottomy], color='red', linewidth=2)
    if ax_rotation:
        midx= startx + (bottomx-startx)//2
        ax.plot( [midx,midx] , [starty,bottomy],color="blue",linewidth=2)
    if scatter:
      ax.scatter([startx],[starty],color="pink",linewidth=3)
      ax.scatter([bottomx],[bottomy],color="orange",linewidth=3)
    plt.show()
  return (startx,bottomx),(starty,bottomy)

def rotate_img(msk, angle=30, plot=True):

    print ('angle= ', angle)
    # Get image size and center
    h, w = msk.shape[:2]
    cx, cy = w // 2, h // 2

    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), -int(angle), 1.0)

    # Apply rotation to image
    rotated = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_LINEAR)
    if plot:
        # Display result
        fig = plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(rotated, cmap="gray")
        plt.title("Rotated Image")
        plt.subplot(1, 2, 2)
        plt.imshow(msk, cmap="gray")
        plt.title("original image")
        plt.show()
    return rotated

def rotation_angle(msk):
    # compute angle

    (startx, bottomx), (starty, bottomy) = fitted_line(msk, plot=True)
    # Calculate the angle of the fitted line
    dy = bottomy - starty
    dx = bottomx - startx
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Calculate the angle difference between the fitted line and vertical axis
    angle_diff = angle - 90
    print(-angle_diff)
    return -angle_diff
def apply_rotate_img(msk, replicate=False):
    alpha = rotation_angle(msk)
    if replicate:
        return alpha
    else:
        return (rotate_img(msk, alpha, False))

def replicate_rotate(img, msk):
    alpha = apply_rotate_img(msk, True)

    return (rotate_img(img, alpha, False))

def result ():
    # plot results
    collected_masks=[]
    for i in range(len(data)):

      original = data[i].copy()
      mask = loaded_model.predict(np.expand_dims(original,axis=0))

      filtered_mask = remove_outliers(mask[0], min_size=100)
      mask = dilation(mask[0])

      segmented = np.squeeze(original).copy()
      segmented[np.squeeze(mask )<0.3] = 0

      filtered_segmented=np.squeeze(original).copy()
      filtered_segmented[np.squeeze(filtered_mask) <0.3] = 0
      rotate_img = apply_rotate_img(mask)
      rotated_sgm = replicate_rotate(filtered_segmented, rotate_img)

      fig = plt.figure(figsize=(8,6))

      plt.subplot(1,2,1)
      plt.imshow(np.squeeze(original))
      plt.title("image"+str(i))
      plt.axis("off")

      plt.subplot(1,2,2)
      plt.imshow(rotated_sgm,cmap="gray")
      plt.title("salamandre"+str(i))
      plt.axis("off")


      fig.tight_layout()
      plt.show()
      figure_file_name="segmented_img"+str(i)+".png"
      collected_masks.append(mask)
      plt.imsave("../content/prediction_img/"+figure_file_name,segmented)

if __name__ == '__main__':
    result()
