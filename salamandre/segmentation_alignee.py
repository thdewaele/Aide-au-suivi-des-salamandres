import os
import cv2
import math
import random
import sklearn
import numpy as np
import pandas as pd
from PIL import Image
#from google.colab import drive
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from sklearn.cluster import KMeans
from skimage.transform import rotate
from pillow_heif import register_heif_opener
from sklearn.metrics import silhouette_score
from skimage.measure import label, regionprops, regionprops_table

images =["/content/drive/MyDrive/upright_salamandre_data/train/images/"+ doc for doc in os.listdir("/content/drive/MyDrive/upright_salamandre_data/train/images")]
v_images= ["/content/drive/MyDrive/upright_salamandre_data/val/images/"+ doc for doc in os.listdir("/content/drive/MyDrive/upright_salamandre_data/val/images")]

masks =["/content/drive/MyDrive/upright_salamandre_data/train/masks/"+doc for doc in os.listdir("/content/drive/MyDrive/upright_salamandre_data/train/masks")]
v_masks=["/content/drive/MyDrive/upright_salamandre_data/val/masks/"+doc for doc in os.listdir("/content/drive/MyDrive/upright_salamandre_data/val/masks")]

img_root="/content/drive/MyDrive//upright_salamandre_data/train/images"
msk_root="/content/drive/MyDrive//upright_salamandre_data/train/masks"

vimg_root="/content/drive/MyDrive//upright_salamandre_data/val/images"
vmsk_root="/content/drive/MyDrive//upright_salamandre_data/val/masks"

pred_images = ["/content/drive/MyDrive/upright_salamandre_data/Boleil/prediction_img/"+ doc for doc in os.listdir("/content/drive/MyDrive/upright_salamandre_data/Boleil/prediction_img")]

pred_masks = ["/content/drive/MyDrive/upright_salamandre_data/Boleil/prediction_msk/"+ doc for doc in os.listdir("/content/drive/MyDrive/upright_salamandre_data/Boleil/prediction_msk")]


pred_root="/content/drive/MyDrive//upright_salamandre_data/Boleil"

pred_images.sort()
pred_masks.sort()

masks.sort()
images.sort()
v_masks.sort()
v_images.sort()


#### FUNCTION CELL ####

def non_stretching_resize(img,cmap,desired_size=256):

    ##NEW CODE
    if isinstance(img,np.ndarray):
      old_size=img.shape
    else:
      old_size= img.size
    ratio= float(desired_size)/max(old_size)
    new_size =tuple([int(x * ratio) for x in old_size])
    #print(new_size)
    im = img.resize(new_size,resample= Image.ANTIALIAS)

    new_im = Image.new(cmap, (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))

    img_array=np.asarray(new_im)/255.
    return new_im

def segment(img,msk):
  image=img.copy()
  image[msk<0.3]=0
  return image

def load_batch(batch=0):
  msk_samples=[]
  img_samples=[]
  sgmnt_samples=[]

  start=50*batch
  for i in range(start,start+50):
    if i>len(masks):
      return msk_samples,img_samples,sgmnt_samples
    msk_= np.asarray(Image.open(os.path.join(msk_root,masks[i])).convert('L'))
    msk_samples.append(msk_)
    img_= np.asarray(Image.open(os.path.join(img_root,images[i])))
    img_samples.append(img_)

    if(msk_.shape != img_.shape[:-1]):
      print(msk_.shape,img_.shape)
      plt.subplot(1,2,1)
      plt.imshow(img_)
      plt.subplot(1,2,2)
      plt.imshow(msk_,cmap="gray")
      plt.show()
    sgmnt_samples.append(segment(img_,msk_))
    print("\r"+str(i)+"/"+str(len(images)),end="")

  return msk_samples,img_samples,sgmnt_samples

def load_pred(batch=0):
  pred_msk_samples=[]
  pred_sgmnt_samples=[]
  start = 50 * batch
  end=start + 50

  if end>len(pred_masks):
   end=len(pred_masks)

  for i in range(batch*50,batch*50+50):

    msk_= np.asarray(Image.open(os.path.join(pred_root,pred_masks[i])).convert('L'))

    threshold_value = 128
    msk_ = (msk_ > threshold_value).astype(int)

    pred_msk_samples.append(msk_)
    img_= np.asarray(Image.open(os.path.join(pred_root,pred_images[i])))
    pred_sgmnt_samples.append(img_)
    print("\r"+str(i)+"/"+str(len(images)),end="")
  return pred_msk_samples,pred_sgmnt_samples

def cropping_bounds(img):
  """
    get coordinates of bounding box around the salamander mask
    img: binary mask of salamander body
    return: quadruplet of bbox coordinates
  """
  regions = regionprops(img)
  props=regions[0]
  minr, minc, maxr, maxc = props.bbox
  """
  if minr>10:
    minr=minr-10
  if minc>10:
    minc=minc-10
  if maxc<250:
    maxc=maxc+10
  if maxr<250:
    maxr=maxr+10
    """
  return minr,minc,maxc,maxr

def crop_image(img):
  """
    crop the mask based on the anontated salamander body bbox
    img: binary mask of salamander body
    return: cropped mask around the animal body (rectangle)

  """
  minr,minc,maxc,maxr=cropping_bounds(img)
  cropped=img[minr:maxr,minc:maxc]
  return cropped

def crop_image_s_mask(img,msk):
  """
    crop the image based on the annotated mask salamander body bbox
    img: image sample
    msk: correcpsonding binary mask
    return: cropped imaged around animal body (rectangle)

  """
  minr,minc,maxc,maxr=cropping_bounds(msk)
  cropped_img=img[minr:maxr,minc:maxc]
  return cropped_img

def zoom_bounds(img,msk,plot=True,angle=False):
  """
    zoom on the image based on the annotated mask salamander body bbox
    img: image sample
    msk: correcpsonding binary mask
    return: cropped imaged around animal body (rectangle)
  """
  SIZE=256
  if not angle: ##NEW CODE
    minr,minc,maxc,maxr=cropping_bounds(msk)
  else:
    minr,minc,maxc,maxr=compute_rotated_bbox(cropping_bounds(msk),apply_rotate_img(msk,True))

  newminr, newminc, newmaxc,newmaxr=minr,minc,maxc,maxr
  h=maxr-minr
  w=maxc-minc

  bound=SIZE
  if img.shape==(SIZE,SIZE):
    bound=max(h,w)
  if h<=SIZE:

    halfr= minr + h//2
    newminr=max( halfr - bound/2,0)
    newmaxr= halfr + bound/2
  else:
    bound=h
    newminr,newmaxr = minr, maxr

  if w<=h:
    halfc= minc+ w//2
    newminc = max(halfc - bound/2,0)
    newmaxc = min(halfc + bound/2,img.shape[1])


  else:
    newminc, newmaxc= minc,maxc

  cropped_img=img[int(newminr):int(newmaxr),int(newminc):int(newmaxc)]
  if plot:
    plt.imshow(cropped_img)
    plt.show()

  return newminr, newminc, newmaxc,newmaxr

def fitted_line(msk_sample,plot=True,scatter=False,ax_rotation=False):
  """
      computed fitted line that goes through salamander body mask
      msk: the binary mask of the salamander body
      plot: boolean, true if we want to splot the result
      return: the coordinates of line start and line end
  """
  contours, hierarchy = cv2.findContours(msk_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  rows, cols = msk_sample.shape
  if not contours: ####NEW CODE
    print("No contours found.")
    plt.imshow(msk_sample,cmap="gray")
    plt.show()
    return None, None
  contours = sorted(contours, key=cv2.contourArea, reverse=True) ###NEW CODE

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

def rotate_img(msk,angle=30,plot=True):

  # Get image size and center
  h, w = msk.shape[:2]
  cx, cy = w // 2, h // 2

  # Calculate rotation matrix
  M = cv2.getRotationMatrix2D((cx, cy), -int(angle), 1.0)

  # Apply rotation to image
  rotated = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_LINEAR)
  if plot:
    # Display result
    fig=plt.figure(figsize=(10,7))
    plt.subplot(1,2,1)
    plt.imshow( rotated,cmap="gray")
    plt.title("Rotated Image")
    plt.subplot(1,2,2)
    plt.imshow(msk,cmap="gray")
    plt.title("original image")
    plt.show()
  return rotated

def rotation_angle(msk):

  #compute angle

  (startx,bottomx),(starty,bottomy)= fitted_line(msk,plot=False)
  # Calculate the angle of the fitted line
  dy = bottomy - starty
  dx = bottomx - startx
  angle = np.arctan2(dy, dx) * 180 / np.pi

  # Calculate the angle difference between the fitted line and vertical axis
  angle_diff = angle - 90

  return -angle_diff

def apply_rotate_img(msk,replicate=False):

  alpha=rotation_angle(msk)
  if replicate:
    return alpha
  else:
    return(rotate_img(msk,alpha,False))

def replicate_rotate(img,msk):

  alpha=apply_rotate_img(msk,True)

  return(rotate_img(img,alpha,False))

# Translation and centering

def center_shift(msk):

  start,end=fitted_line(msk,False)

  # Calculate the center of the bounding box
  bbox_center = np.array([start[0], (end[1]//2)])

  h, w = msk.shape[:2]

  # Calculate the center of the image
  mskcenter = np.array([w/2, h/2])

  # Calculate the shift needed to move the bounding box to the center of the image
  shift = mskcenter - bbox_center
  # Define the translation matrix
  translation_matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)

  return translation_matrix


def center_mask(msk,replicate=False):

  translation_matrix=center_shift(msk)
  h, w = msk.shape[:2]
  msk_shifted = cv2.warpAffine(msk, translation_matrix, (w, h))
  if replicate:
    return translation_matrix
  else:
    return msk_shifted

def replicate_centering(img,msk):

  translation_matrix=center_mask(msk,True)
  h, w = img.shape[:2]
  img_shifted = cv2.warpAffine(img, translation_matrix, (w, h))
  return img_shifted

def run_affine_transform(msk,img=None,replicate=False):
  """
    applied affine transformation to match desired template
    msk: binary mask to transform
    img: corresponding segmented image, used if replicate= True
    replicate: boolean, if false applies transformation to msk, if true maps msk transformation to img
    return: transformed mask (resize, rotate, scale, translate)
  """

  #resize to 256,256
  to_resize = Image.fromarray((msk * 255).astype(np.uint8))
  sample= np.array(non_stretching_resize(to_resize,"L"))

  # rotate image
  start,end=fitted_line(sample,False)
  rotated_img=apply_rotate_img(sample)

  # zoom in and crop
  newminr, newminc, newmaxc,newmaxr=zoom_bounds(rotated_img,rotated_img,False)
  cropped_img=rotated_img[int(newminr):int(newmaxr),int(newminc):int(newmaxc)]

  # translate
  img_shifted = center_mask(cropped_img)

  if replicate:
    #print("replicate on sgmnt")
    to_resize2 = Image.fromarray((img * 1).astype(np.uint8))
    segm= np.array(non_stretching_resize(to_resize2,"RGB"))
    rotated_sgm=replicate_rotate(segm,sample)
    cropped_sgm=rotated_sgm[int(newminr):int(newmaxr),int(newminc):int(newmaxc)]
    sgm_shifted = replicate_centering(cropped_sgm,cropped_img)

    return img_shifted,sgm_shifted
  else:
    return img_shifted


def curved_salam(msk,return_angle=False):
  """
    returns if the salamander if curved or not
    msk: salamander mask
    return: boolean that is true
  """
  THRESHOLD=165
  h,c,t= approx_polygone_points(msk)
  x1, y1 = h[0][0], h[0][1]
  x2, y2 = cx, cy= c
  upper_body_vec = np.array([h[0][0] - x2, h[0][1] - y2])
  lower_body_vec = np.array([t[0][0] - x2, t[0][1] - y2])
  angle = np.degrees(np.arctan2(np.linalg.det([upper_body_vec, lower_body_vec]), np.dot(upper_body_vec, lower_body_vec)))
  if return_angle:
    return abs(angle)<THRESHOLD,angle
  else:
    return abs(angle)<THRESHOLD

def approx_polygone_points(msk,plot=False):

  contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnt=np.concatenate(contours)
  perimeter = cv2.arcLength(cnt, True)

    # Approximate the contour with a polygonal curve
  approx_curve = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    # Centroid
  M = cv2.moments(cnt)
  cx = int(M['m10']/M['m00'])
  cy = int(M['m01']/M['m00'])
  centroid = (cx,cy)

  img = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
  approx_curve = np.append(approx_curve, [approx_curve[0]], axis=0)

  x = approx_curve[:, 0, 0]
  y = approx_curve[:, 0, 1]
  head_point=min(approx_curve,key=lambda x: x[0][1])
  tail_point=max(approx_curve,key=lambda x: x[0][1])
  if plot:
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

    img = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
    axs[0].imshow(img)
    axs[0].scatter(head_point[0][0],head_point[0][1],color="orange",linewidth=2.5,zorder=3,label="head point")
    axs[0].scatter(tail_point[0][0],tail_point[0][1],color="green",linewidth=2.5,zorder=3,label="tail point")
    axs[0].scatter(cx,cy,color="pink",linewidth=2.5,zorder=3,label="centroid point")

    axs[0].plot(x, y, '-r', linewidth=2,zorder=2)
    axs[0].plot([head_point[0][0],cx],[head_point[0][1],cy],color="orange",linewidth=2)#upper body line
    axs[0].plot([tail_point[0][0],cx],[tail_point[0][1],cy],color="green",linewidth=2) #lower body line
    axs[1].imshow(msk,cmap="gray")
    upper_body_vec = np.array([head_point[0][0] - cx, head_point[0][1] - cy])
    lower_body_vec = np.array([tail_point[0][0] - cx, tail_point[0][1] - cy])
    angle = np.degrees(np.arctan2(np.linalg.det([upper_body_vec, lower_body_vec]), np.dot(upper_body_vec, lower_body_vec)))
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
      # print("INSIDE UNWARP: centroid={} ; head_point= {} ; tail_point= {} ".format(centroid,head_point,tail_point))
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