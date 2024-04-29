import os
import random

def rename_set(folder):

    for letter in os.listdir(folder):
        
        folder_letter = os.path.join(folder, letter)
        print('Renaming Images of Letter ' + letter)
        i=0
        for image in os.listdir(folder_letter):
            
            renamed_img = os.path.join(folder_letter, '{}_{}.jpg'.format(letter, i))
            img = os.path.join(folder_letter, image)
            
            os.rename(img, renamed_img)
            i+=1
            

folder_train = os.path.join(os.path.dirname(__file__), 'data/syntetic_asl_dataset/Train_Alphabet')
folder_test = os.path.join(os.path.dirname(__file__), 'data/syntetic_asl_dataset/Test_Alphabet')

print('RENAMING TRAIN-SET...')
rename_set(folder_train)
print('RENAMING TEST-SET...')
rename_set(folder_test)