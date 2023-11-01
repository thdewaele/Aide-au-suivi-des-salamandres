import cv2
import numpy as np

def image_moment(img):
   image = cv2.imread(img)
   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray, (5, 5), 0)
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
   print(a,b)
   tab_is_tache = [[0 for j in range(b)] for i in range(a)]

   distance_seuil = 1
   # Vérification des contours pour chaque case du tableau
   for i, ligne in enumerate(tab_coordonnees):
      for j, case in enumerate(ligne):
         x, y = case
         case_contour = False  # Initialisation du drapeau de contour
         for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= distance_seuil:
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






   for ligne in tab_is_tache:
      print(ligne)
   cv2.imshow('Contour de la forme', image_contour)
   cv2.waitKey(0)
   cv2.destroyAllWindows()


if __name__ == '__main__':
    image_moment('1005.jpg')