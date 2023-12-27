
import os
import matplotlib
import matplotlib.pyplot as plt

#from Segmentation.Plots import *
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageOps,Image
#import tensorflow_datasets as tfds
from pillow_heif import register_heif_opener
from skimage.measure import regionprops
from tensorflow.keras import Model,optimizers,layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, load_img
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,Dropout,Concatenate,Input
from skimage.transform import rotate
DESIRED_SIZE = 256

def dice_coef(y_true, y_pred,smooth = 10.):
    """ Dice's coefficient
    param:
        y_true: correct mask of the salamander
        y_pred: predicted mask of the salamander
    """

   # print("y_true shape {}".format(y_true.shape))
    y_true_f = K.flatten(y_true)
   # print("y_pred shape {}".format(y_pred.shape))

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


def double_conv_block(prev_layer, filter_count):
    """ double_conv_block
    param:
        prev_layer: previous layer to connect to the double convolution block
        filter_count: number of filters to build in the convolution layer
    """
    new_layer = Conv2D(filter_count, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(prev_layer)
    new_layer = Conv2D(filter_count, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(new_layer)
    return new_layer

def downsample_block(prev_layer, filter_count):
    """ encoder block
    param:
        prev_layer: previous layer to connect to the double convolution block
        filter_count: number of filters to build in the convolution layer
    """
    skip_features = double_conv_block(prev_layer, filter_count)
    down_sampled = MaxPooling2D(2)(skip_features)
    #down_sampled = Dropout(0.3)(down_sampled)
    return skip_features, down_sampled

def upsample_block(prev_layer, skipped_features, n_filters):
    """ decoder block
    param:
        prev_layer: previous layer to connect to the double convolution block
        filter_count: number of filters to build in the convolution layer
    """
    upsampled = Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding="same")(prev_layer)
    upsampled = Concatenate()([upsampled, skipped_features])
    #upsampled = Dropout(0.3)(upsampled)
    upsampled = double_conv_block(upsampled, n_filters)
    return upsampled

def make_unet():

    inputs = Input(shape=(256, 256, 3))

    skipped_fmaps_1, downsample_1 = downsample_block(inputs, 64)
    skipped_fmaps_2, downsample_2 = downsample_block(downsample_1, 128)
    skipped_fmaps_3, downsample_3 = downsample_block(downsample_2, 256)
    skipped_fmaps_4, downsample_4 = downsample_block(downsample_3, 512)

    bottleneck = double_conv_block(downsample_4, 1024)

    upsample_1 = upsample_block(bottleneck, skipped_fmaps_4, 512)
    upsample_2 = upsample_block(upsample_1, skipped_fmaps_3, 256)
    upsample_3 = upsample_block(upsample_2, skipped_fmaps_2, 128)
    upsample_4 = upsample_block(upsample_3, skipped_fmaps_1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(upsample_4)
    # outputs = Conv2D(2, 1, padding="same", activation = "softmax")(upsample_4)

    unet_model = Model(inputs, outputs, name="U-Net")

    return unet_model


def dataset_paths():
    """
    returns lists of paths to images for each dataset portion i.e training and validation to correctly load dataset.

    """
    images = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/images/" + doc for doc in
              os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/images") if doc !=".DS_Store"]

    v_images = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/images/" + doc for doc in
                os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/images")if doc !=".DS_Store"]


    masks = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/masks/" + doc for doc in
             os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/masks")if doc !=".DS_Store"]

    v_masks = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/masks/" + doc for doc in
               os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/masks")if doc !=".DS_Store"]

    masks.sort()
    images.sort()
    v_masks.sort()
    v_images.sort()

    return images,masks,v_images,v_masks



def make_dataset(images,masks,v_images,v_masks,validation=False):
  """

  :param images: list of paths to training images
  :param masks: list of paths to validation images
  :param v_images: list of paths to validation images
  :param v_masks: list of paths to validation masks
  :type  validation: bool
  :return: numpy arrays of images and their corresponding masks (training set or validation set)
  """
  x = []
  y = []
  if(validation):
    for i,(image,mask) in enumerate(zip(v_images[:10000],v_masks[:10000])):
      if image==".DS_Store":
        print('found the impostor')
        pass
      print("\r"+str(i)+"/"+str(len(v_images)),end="")

      image = Image.open(os.path.join("/content/drive/MyDrive//upright_salamandre_data/val/images",image))#.convert('L')
      mask = Image.open(os.path.join("/content/drive/MyDrive//upright_salamandre_data/val/masks",mask)).convert('L')

      image = np.asarray(image.resize((256,256)))/255.
      mask = np.asarray(mask.resize((256,256)))/255.

      x.append(image)
      y.append(mask)
  else:
    for i,(image,mask) in enumerate(zip(images[:10000],masks[:10000])):
      print("\r"+str(i)+"/"+str(len(images)),end="")
      if ".DS_Store" in image:
        print('found the impostor')
        pass
      image = Image.open(os.path.join("/content/drive/MyDrive//upright_salamandre_data/train/images",image))#.convert('L')
      mask = Image.open(os.path.join("/content/drive/MyDrive//upright_salamandre_data/train/masks",mask)).convert('L')

      image = np.asarray(image.resize((256,256)))/255.
      mask = np.asarray(mask.resize((256,256)))/255.

      x.append(image)
      y.append(mask)

  return np.array(x),np.array(y)



def get_train_generator(x,y,seed=1):
    data_gen_args = dict(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow(x,batch_size=32,shuffle=True,seed=seed)
    mask_generator = mask_datagen.flow(y,batch_size=32,shuffle=True,seed=seed)
    train_generator = zip(image_generator, mask_generator)

    return train_generator

def get_val_generator(v_x,v_y,seed=1):

    image_test_datagen = ImageDataGenerator()
    mask_test_datagen = ImageDataGenerator()

    image_test_generator = image_test_datagen.flow(v_x,batch_size=32,seed=seed)

    mask_test_generator = mask_test_datagen.flow(v_y,batch_size=32,seed=seed)

    valid_generator = zip(image_test_generator, mask_test_generator)

    return valid_generator

def present_data():

    images, masks, v_images, v_masks=dataset_paths()
    register_heif_opener()
    x,y= make_dataset(images,masks,v_images,v_masks)
    v_x,v_y = make_dataset(images,masks,v_images,v_masks,True)
    y =np.expand_dims(y,axis=-1)
    v_y =np.expand_dims(v_y,axis=-1)

    return x,y,v_x,v_y


def remove_outliers(mask, min_size=100):#from notebook "segmented_model"
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


def dilation(msk): #from notebook "segmented_model"

    msk = (msk * 255).astype(np.uint8)

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(msk, disk)

    return img

def non_stretching_resize(img ,cmap ,desired_size=512):

    old_size= img.size
    ratio= float(desired_size ) /max(old_size)
    new_size =tuple([int(x * ratio) for x in old_size])
    im = img.resize(new_size, resample=Image.LANCZOS)#Image.ANTIALIAS)

    new_im = Image.new(cmap, (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))

    img_array = np.asarray(new_im) / 255.
    return new_im
def run_trained_model(path_model=None,path_image=None,return_result=False, plot_result=False):
    """
        run_trained_model runs the inference of the trained model on the chosen image.
        path_model: if the parameter is provided the model is loaded from that path; if path_model is set to None the default model is loaded
        path_image:if the parameter is provided the image is loaded from that path; if path_image is set to None the default image is loaded
        return_result: if False the segmented salamander is displayed but not returned; if True the function returns the predicted mask and segmented body
    """
    print("Je passe de run_trained_model")

    # Load the saved model and specify the custom loss function
    path_model = "salamandre/u_net_upright_data-2.h5"
    if path_model == None:
        default_model_path="u_net_upright_data-2.h5"
        with tf.keras.utils.custom_object_scope({'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}):
            loaded_model = load_model(default_model_path)
    else:
        with tf.keras.utils.custom_object_scope({'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}):
            loaded_model = load_model(path_model)
    #load the image
    """
    if path_image==None:
        default_image_path="../images/003.JPG"
        original = Image.open(default_image_path)
        original = np.asarray(non_stretching_resize(original, "RGB",256)) / 255.
        original = np.array(original)
        print(original.shape)
    else:
        original = Image.open(path_image)
        original = np.asarray(non_stretching_resize(original, "RGB",256)) / 255.
        original = np.array(original)
        
    """
    original = Image.open(path_image)
    original = np.asarray(non_stretching_resize(original, "RGB",256)) / 255.
    original = np.array(original)

    mask = loaded_model.predict(np.expand_dims(original, axis=0))

    filtered_mask = remove_outliers(mask[0], min_size=100)
    mask = dilation(mask[0])

    segmented = np.squeeze(original).copy()
    segmented[np.squeeze(mask) < 0.3] = 0
    binary_mask = (mask > 148).astype(np.uint8)

    filtered_segmented = np.squeeze(original).copy()
    filtered_segmented[np.squeeze(filtered_mask) < 0.3] = 0


    #print(" original image shape {} \n segmented image shape {} \n mask shape {} \n binary_mask shape {}".format(
     #   original.shape, segmented.shape, mask.shape, binary_mask.shape))

    ##### Plot the results like in the notebook
    if (plot_result):
        plt.imshow(filtered_segmented,cmap="gray")
        plt.show()

        fig = plt.figure(figsize=(8, 6))

        plt.subplot(1, 4, 1)
        plt.imshow(np.squeeze(original))
        plt.title("image")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(filtered_segmented, cmap="gray")
        plt.title("salamandre")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(binary_mask, cmap="gray")
        plt.title("binary mask")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(mask, cmap="gray")
        plt.title("mask")
        plt.axis("off")

        fig.tight_layout()
        plt.show()
        plt.close()

    if return_result:
        return binary_mask,filtered_segmented

def cropping_bounds(img):
    """
      get coordinates of bounding box around the salamander mask
      img: binary mask of salamander body
      return: quadruplet of bbox coordinates
    """
    regions = regionprops(img)
    props = regions[0]
    minr, minc, maxr, maxc = props.bbox

    return minr, minc, maxc, maxr

def crop_image(img):
    """
      crop the mask based on the annotated salamander body bbox
      img: binary mask of salamander body
      return: cropped mask around the animal body (rectangle)

    """
    minr, minc, maxc, maxr = cropping_bounds(img)
    cropped = img[minr:maxr, minc:maxc]
    return cropped

def crop_image_s_mask(img, msk):
    """
      crop the image based on the annotated mask salamander body bbox
      img: image sample
      msk: corresponding binary mask
      return: cropped imaged around animal body (rectangle)

    """
    minr, minc, maxc, maxr = cropping_bounds(msk)
    cropped_img = img[minr:maxr, minc:maxc]
    return cropped_img

def zoom_bounds(img, msk, plot=True):
    """
      zoom on the image based on the annotated mask salamander body bbox
      img: image sample
      msk: corresponding binary mask
      return: cropped imaged around animal body (rectangle)
    """
    SIZE = DESIRED_SIZE  # 256
    minr, minc, maxc, maxr = cropping_bounds(msk)
    newminr, newminc, newmaxc, newmaxr = minr, minc, maxc, maxr
    h = maxr - minr
    w = maxc - minc

    bound = SIZE
    if img.shape == (SIZE, SIZE):
        bound = max(h, w)
    if h <= SIZE:

        halfr = minr + h // 2
        newminr = max(halfr - bound / 2, 0)
        newmaxr = halfr + bound / 2
    else:
        bound = h
        newminr, newmaxr = minr, maxr

    if w <= h:
        halfc = minc + w // 2
        newminc = max(halfc - bound / 2, 0)
        newmaxc = min(halfc + bound / 2, img.shape[1])


    else:
        newminc, newmaxc = minc, maxc

    cropped_img = img[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]
    if plot:
        plt.imshow(cropped_img)
        plt.show()

    return newminr, newminc, newmaxc, newmaxr

def fitted_line(msk_sample, plot=True, scatter=False, ax_rotation=False):
    """
        computed fitted line that goes through salamander body mask
        msk: the binary mask of the salamander body
        plot: boolean, true if we want to splot the result
        return: the coordinates of line start and line end
    """
    contours, hierarchy = cv2.findContours(msk_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows, cols = msk_sample.shape

    if not contours:  #### added verification
        print("No contours found.")
        plt.imshow(msk_sample, cmap="gray")
        plt.show()
        return None, None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  ###added verification

    # Calculate endpoints of the line
    vx, vy, x, y = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)

    starty = 0

    if vy != 0:
        startx = int((starty - y) * (vx / vy) + x)
        bottomy = rows - 1
        bottomx = int((bottomy - y) * (vx / vy) + x)
    else:
        startx = bottomx = x
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
            midx = startx + (bottomx - startx) // 2
            ax.plot([midx, midx], [starty, bottomy], color="blue", linewidth=2)
        if scatter:
            ax.scatter([startx], [starty], color="pink", linewidth=3)
            ax.scatter([bottomx], [bottomy], color="orange", linewidth=3)
        plt.show()
    return (startx, bottomx), (starty, bottomy)

def rotate_img(msk, angle=30, plot=True):
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

    (startx, bottomx), (starty, bottomy) = fitted_line(msk, plot=False)
    # Calculate the angle of the fitted line
    dy = bottomy - starty
    dx = bottomx - startx
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Calculate the angle difference between the fitted line and vertical axis
    angle_diff = angle - 90

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

def center_shift(msk):
    start, end = fitted_line(msk, False)

    # Calculate the center of the bounding box
    bbox_center = np.array([start[0], (end[1] // 2)])

    h, w = msk.shape[:2]

    # Calculate the center of the image
    mskcenter = np.array([w / 2, h / 2])

    # Calculate the shift needed to move the bounding box to the center of the image
    shift = mskcenter - bbox_center
    # Define the translation matrix
    translation_matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)

    return translation_matrix

def center_mask(msk, replicate=False):
    translation_matrix = center_shift(msk)
    h, w = msk.shape[:2]
    msk_shifted = cv2.warpAffine(msk, translation_matrix, (w, h))
    if replicate:
        return translation_matrix
    else:
        return msk_shifted

def replicate_centering(img, msk):
    translation_matrix = center_mask(msk, True)
    h, w = img.shape[:2]
    img_shifted = cv2.warpAffine(img, translation_matrix, (w, h))
    return img_shifted

def run_affine_transform(msk, img=None, replicate=False):
    """
      applied affine transformation to match desired template
      msk: binary mask to transform
      img: corresponding segmented image, used if replicate= True
      replicate: boolean, if false applies transformation to msk, if true maps msk transformation to img
      return: transformed mask (resize, rotate, scale, translate)
    """

    # resize to 256,256
    to_resize = Image.fromarray((msk * 255).astype(np.uint8))
    sample = np.array(non_stretching_resize(to_resize, "L"))

    # rotate image
    start, end = fitted_line(sample, False)
    rotated_img = apply_rotate_img(sample)

    # zoom in and crop
    newminr, newminc, newmaxc, newmaxr = zoom_bounds(rotated_img, rotated_img, False)
    cropped_img = rotated_img[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]

    # translate
    img_shifted = center_mask(cropped_img)

    if replicate:
        # print("replicate on sgmnt")
        to_resize2 = Image.fromarray((img * 1).astype(np.uint8))
        segm = np.array(non_stretching_resize(to_resize2, "RGB"))
        rotated_sgm = replicate_rotate(segm, sample)
        cropped_sgm = rotated_sgm[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]
        sgm_shifted = replicate_centering(cropped_sgm, cropped_img)

        return img_shifted, sgm_shifted
    else:
        return img_shifted

def curved_salam(msk, return_angle=False):
    """
      returns if the salamander if curved or not
      msk: salamander mask
      return: boolean that is true
    """
    THRESHOLD = 165
    h, c, t = approx_polygone_points(msk)
    x1, y1 = h[0][0], h[0][1]
    x2, y2 = cx, cy = c
    upper_body_vec = np.array([h[0][0] - x2, h[0][1] - y2])
    lower_body_vec = np.array([t[0][0] - x2, t[0][1] - y2])
    angle = np.degrees \
        (np.arctan2(np.linalg.det([upper_body_vec, lower_body_vec]), np.dot(upper_body_vec, lower_body_vec)))
    if return_angle:
        return abs(angle) < THRESHOLD, angle
    else:
        return abs(angle) < THRESHOLD

def approx_polygone_points(msk, plot=False):

    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.concatenate(contours)
    perimeter = cv2.arcLength(cnt, True)

    # Approximate the contour with a polygonal curve
    approx_curve = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    # Centroid
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    centroid = (cx, cy)

    img = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
    approx_curve = np.append(approx_curve, [approx_curve[0]], axis=0)

    x = approx_curve[:, 0, 0]
    y = approx_curve[:, 0, 1]
    head_point = min(approx_curve, key=lambda x: x[0][1])
    tail_point = max(approx_curve, key=lambda x: x[0][1])
    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

        img = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
        axs[0].imshow(img)
        axs[0].scatter(head_point[0][0], head_point[0][1], color="orange", linewidth=2.5, zorder=3,
                       label="head point")
        axs[0].scatter(tail_point[0][0], tail_point[0][1], color="green", linewidth=2.5, zorder=3,
                       label="tail point")
        axs[0].scatter(cx, cy, color="pink", linewidth=2.5, zorder=3, label="centroid point")

        axs[0].plot(x, y, '-r', linewidth=2, zorder=2)
        axs[0].plot([head_point[0][0], cx], [head_point[0][1], cy], color="orange", linewidth=2)  # upper body line
        axs[0].plot([tail_point[0][0], cx], [tail_point[0][1], cy], color="green", linewidth=2)  # lower body line
        axs[1].imshow(msk, cmap="gray")
        upper_body_vec = np.array([head_point[0][0] - cx, head_point[0][1] - cy])
        lower_body_vec = np.array([tail_point[0][0] - cx, tail_point[0][1] - cy])
        angle = np.degrees \
            (np.arctan2(np.linalg.det([upper_body_vec, lower_body_vec]), np.dot(upper_body_vec, lower_body_vec)))
        print("Angle between upper and lower body lines: {:.2f} degrees".format(abs(angle)))

        plt.show()

    return head_point, centroid, tail_point

def unwrap_curve(msk, sgm, normalized=False, smooth=0):

    # normalize if necessary
    if not (normalized):
        msk, sgm = run_affine_transform(msk, sgm, replicate=True)

    sgm = cv2.copyMakeBorder(sgm, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value=0)
    msk = cv2.copyMakeBorder(msk, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value=0)

    # get required points
    head_point, centroid, tail_point = approx_polygone_points(msk)
    x1, y1 = head_point[0][0], head_point[0][1]
    x2, y2 = cx, cy = centroid[0], centroid[1]
    # print("INSIDE UNWARP: centroid={} ; head_point= {} ; tail_point= {} ".format(centroid ,head_point ,tail_point))
    # computer angle between upper body vect and vertical axis
    angle = np.arctan2(x2 - x1, y2 - y1) * 180 / np.pi

    # split segment int upper and lower based on centroid
    upper_bod = sgm[:cy, :]
    lower_bod = sgm[cy:, :]
    upper_bod_m = msk[:cy, :]
    lower_bod_m = msk[cy:, :]
    # rotate uppder bod
    rotated_upper_bod = rotate(upper_bod, -angle)
    rotated_upper_bod = (rotated_upper_bod * 255).astype(np.uint8)
    rotated_upper_bod_m = rotate(upper_bod_m, -angle)
    rotated_upper_bod_m = (rotated_upper_bod_m * 255).astype(np.uint8)

    # trnaslate lower bod

    shiftx = cx - head_point[0][0]
    if shiftx > 0:
        # print("to the left")
        shiftx = shiftx - smooth
    else:
        shiftx = shiftx + smooth
        # print("to the right")

    M = np.float32([[1, 0, -(shiftx)],
                    [0, 1, 0]])
    rows, cols, ch = lower_bod.shape
    # print(shiftx)

    lower_bod = cv2.warpAffine(lower_bod, M, (cols, rows))
    lower_bod_m = cv2.warpAffine(lower_bod_m, M, (cols, rows))

    # concatenate upper and lower
    new_salam_sgm = np.concatenate((rotated_upper_bod, lower_bod), axis=0)
    new_salam_msk = np.concatenate((rotated_upper_bod_m, lower_bod_m), axis=0)

    return (new_salam_sgm, new_salam_msk)

def sheer(msk, sgm, normalized=False, normalize=True):

    # transform if necessary
    if not (normalized):
        msk, sgm = run_affine_transform(msk, sgm, replicate=True)

    # expand black background in case the foreground moves off the borders
    sgm = cv2.copyMakeBorder(sgm, 120, 10, 120, 0, cv2.BORDER_CONSTANT, None, value=0)
    msk = cv2.copyMakeBorder(msk, 120, 10, 120, 0, cv2.BORDER_CONSTANT, None, value=0)
    # sheer transform
    M = np.float32([[1, -0.7, 0],
                    [0, 1, 0]])
    rows, cols, ch = sgm.shape
    sheer_sgm = cv2.warpAffine(sgm, M, (cols, rows))
    sheer_msk = cv2.warpAffine(msk, M, (cols, rows))

    if normalize:
        sheer_msk, sheer_sgm = run_affine_transform(sheer_msk, sheer_sgm, True)

    return sheer_msk, sheer_sgm



################################### - PLOTS - #######################################
################################### - PLOTS - #######################################
################################### - PLOTS - #######################################
################################### - PLOTS - #######################################

def affine_transform_msk_sgmnt(msk, sgm):
    """
    affine_transform_msk_sgmnt runs the affine transformation operations on msk and sgm and returns the resulting images
    """
    # rotate image

    rotated_msk = apply_rotate_img(msk)
    rotated_sgm = replicate_rotate(sgm, msk)

    # zoom in and crop
    newminr, newminc, newmaxc, newmaxr = zoom_bounds(rotated_msk, rotated_msk, False)
    cropped_msk = rotated_msk[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]
    cropped_sgm = rotated_sgm[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]

    # translate
    msk_shifted = center_mask(cropped_msk)
    sgm_shifted = replicate_centering(cropped_sgm, cropped_msk)

    return msk_shifted, sgm_shifted

def affine_transform_msk_sgmnt(msk, sgm):
    """
    affine_transform_msk_sgmnt runs the affine transformation operations on msk and sgm and returns the resulting images
    """
    print('Je suis là')
    # rotate image
    rotated_msk = apply_rotate_img(msk)
    rotated_sgm = replicate_rotate(sgm, msk)

    # zoom in and crop
    newminr, newminc, newmaxc, newmaxr = zoom_bounds(rotated_msk, rotated_msk, False)
    cropped_msk = rotated_msk[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]
    cropped_sgm = rotated_sgm[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]

    # translate
    msk_shifted = center_mask(cropped_msk)
    sgm_shifted = replicate_centering(cropped_sgm, cropped_msk)

    return msk_shifted, sgm_shifted

def segmentation(image):
    msk, sgmnt = run_trained_model(path_image=image, return_result=True)

    msk2, sgmnt2 = affine_transform_msk_sgmnt(msk, sgmnt)
    plt.imsave('image.jpg', sgmnt2)
    return msk2, sgmnt2


if __name__ == '__main__':
    msk, sgmnt = segmentation('../images/003.JPG')
    plt.imshow(msk, cmap="gray")
    plt.show()
    plt.imshow(sgmnt)
    plt.imsave('image.jpg',sgmnt)
    plt.show()