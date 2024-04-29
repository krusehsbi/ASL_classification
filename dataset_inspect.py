import cv2
import os
import random


folder = os.path.join(os.path.dirname(__file__), 'data/syntetic_asl_dataset/Test_Alphabet')

for letter in os.listdir(folder):
    folder_letter = os.path.join(folder, letter)
    all_letters = os.listdir(folder_letter)
    for image in random.sample(all_letters,1):
        img = os.path.join(folder_letter, image)

        im = cv2.imread(img)
        cv2.imshow(letter, im)
        cv2.waitKey(0)
  
cv2.destroyAllWindows()
