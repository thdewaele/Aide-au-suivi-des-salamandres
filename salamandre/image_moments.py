
import cv2
import numpy as np
from Segmentation import segmentation

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
   cv2.imshow('tresh',tresh)
   contours, _ = cv2.findContours(tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   for contour in contours:
      M = cv2.moments(contour)
      if M["m00"] != 0:
         cx = int(M["m10"] / M["m00"])
         cy = int(M["m01"] / M["m00"])

   image_contour = cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
   #quadrillage sur l'image pour récuperer les positions
   largeur_image, hauteur_image, _ = image.shape
   espacement_horizontal = 15
   espacement_vertical = 15
   couleur_lignes = (255, 255, 0)
   epaisseur_lignes = 1
   tab_coordonnees =[]

   for y in range(0, hauteur_image, espacement_vertical):
      ligne = []
      cv2.line(image, (0, y), (largeur_image, y), couleur_lignes, epaisseur_lignes)
      for x in range(0, largeur_image, espacement_horizontal):
         cv2.line(image, (x, 0), (x, hauteur_image), couleur_lignes, epaisseur_lignes)
         ligne.append((x,y))
      tab_coordonnees.append(ligne)

   a = len(tab_coordonnees)
   b = len(tab_coordonnees[0])

   tab_is_tache = [[0 for j in range(b)] for i in range(a)]

   distance_seuil = 0.5

   for i, ligne in enumerate(tab_coordonnees):
      for j, case in enumerate(ligne):
         x, y = case
         case_contour = False
         for contour in contours:
            if cv2.pointPolygonTest(contour, (x+5, y+5), False) >= distance_seuil:
               case_contour = True
               break
         if case_contour:
            tab_is_tache[i][j]=1

         else:
            tab_is_tache[i][j]=0

   for i in range (a-1):
      for j in range(b-1):
         if (tab_is_tache[i][j]==1):
            if (tab_is_tache[i+1][j]==1):
               tab_is_tache[i][j]=2
            elif(tab_is_tache[i][j+1]==1):
               tab_is_tache[i][j]=2








   cv2.imshow('Contour de la forme', image_contour)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   return tab_is_tache, tab_coordonnees

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


if __name__ == '__main__':
   msk,sgmt = segmentation("../images/003.JPG")

   tab1, tabcoor = image_moment("image.jpg",0)









