
import cv2
import numpy as np
from salamandre.Segmentation  import segmentation

def image_moment(img,ero):

   image = cv2.imread(img)
   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   kernel = np.ones((5, 5), np.uint8)
   if (ero):
      erosion = cv2.erode(gray, kernel, iterations=1)
   else:
      erosion = gray
   blur = cv2.GaussianBlur(erosion, (5, 5), 0)
   if (ero):
      ret, tresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
   else:
      ret, tresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)
   tab_taches =[[0 for _ in range(14)] for _ in range (14)]

   cv2.imshow('tresh',tresh)
   contours, _ = cv2.findContours(tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   for contour in contours:
      M = cv2.moments(contour)
      if M["m00"] != 0:
         cx = int(M["m10"] / M["m00"])
         cy = int(M["m01"] / M["m00"])
         case_x = min(cx//8,13)
         case_y = min(cy//8,13)

         tab_taches[case_y][case_x] = 1
   for i in range (13):
      for j in range(13):
         if (tab_taches[i][j]==1):
            if (tab_taches[i+1][j]==1):
               tab_taches[i][j]=2
            elif(tab_taches[i][j+1]==1):
               tab_taches[i][j]=2
   """
   tab_coordonnees =[]
   largeur_image, hauteur_image, _ = image.shape
   espacement_horizontal = 8
   espacement_vertical = 8
   couleur_lignes = (255, 255, 0)
   epaisseur_lignes = 1
   for y in range(0, hauteur_image, espacement_vertical):
      ligne = []
      cv2.line(image, (0, y), (largeur_image, y), couleur_lignes, epaisseur_lignes)
      for x in range(0, largeur_image, espacement_horizontal):
         cv2.line(image, (x, 0), (x, hauteur_image), couleur_lignes, epaisseur_lignes)
         ligne.append((x,y))
      tab_coordonnees.append(ligne)
   image_contour = cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
   cv2.imshow('Contour de la forme', image_contour)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   """
   return tab_taches,
def comptagediff(tab1, tab2):
   compt =0
   list_ind = []
   for i in range (len(tab1)):
      for j in range (len(tab1[0])):
         if (tab1[i][j] != tab2[i][j]):
            compt= compt+1
            tab = [i,j]
            list_ind.append(tab)
   return compt, list_ind
def comparasiontab(tab1, tab2):
   for i in range (len(tab1)):
      for j in range (len(tab1[0])):
         if (tab1[i][j] != tab2[i][j]):
            return 0

   return 1

def fusiontab(tab1, tab2):
   tab3 = [[0 for j in range(len(tab1))] for i in range(len(tab1[0]))]
   for i in range(len(tab1)):
      for j in range(len(tab1[0])):
         if (tab1[i][j] == 1 and tab2[i][j] == 1):
            tab3[i][j] = 1
         elif (tab1[i][j] == 2 and tab2[i][j] == 2):
            tab3[i][j] = 2
         elif (tab1[i][j] == 1 and tab2[i][j] == 2):
            tab3[i][j] = 1
         elif (tab1[i][j] == 2 and tab2[i][j] == 1):
            tab3[i][j] = 1
         else:
            tab3[i][j] = 0
   return tab3

def printable(tab):
   for ligne in tab:
      print(ligne)
      print("\n")


def get_table(image):
   msk, sgmt = segmentation(image)
   tab1 = image_moment("image.jpg", 0)
   return tab1


if __name__ == '__main__':
   msk,sgmt = segmentation("../images/003.JPG")

   tab1= image_moment("image.jpg",0)
   image_moment('../content/prediction_img/003.JPG',0)
   printable(tab1)











