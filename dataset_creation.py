import os
import random
import csv

def format_set(folder):

    directories = os.listdir(folder)

    # Add to CSV
    csv_path = os.path.join(folder, 'data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['path', 'letter'])  # Write header row
        for letter in directories:
            
            folder_letter = os.path.join(folder, letter)
            print('Renaming Images of Letter ' + letter)
            i=0
            for image in os.listdir(folder_letter):
                
                renamed_img = os.path.join(folder_letter, '{}_{}.jpg'.format(letter, i))
                img = os.path.join(folder_letter, image)
                
                # Rename file if not exists
                if not os.path.exists(renamed_img):
                    os.rename(img, renamed_img)
                
                i+=1

                csv_writer.writerow([renamed_img, letter])
            
            

folder_train = os.path.join(os.path.dirname(__file__), 'data/synthetic_asl_dataset/Train_Alphabet')
folder_test = os.path.join(os.path.dirname(__file__), 'data/synthetic_asl_dataset/Test_Alphabet')

print('RENAMING TRAIN-SET...')
format_set(folder_train)
print('RENAMING TEST-SET...')
format_set(folder_test)