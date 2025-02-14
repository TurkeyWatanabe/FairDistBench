import pandas as pd
from PIL import Image
import os
import logging
import pandas as pd
import os
import shutil

import pandas as pd

def excel_to_json(excel_file_path, json_file_path):
    """
    Convert an Excel file to JSON format and save it.

    Parameters:
    excel_file_path (str): Path to the input Excel file.
    json_file_path (str): Path to save the output JSON file.
    Input: Labeled Excel file
    Output: Labeled JSON file
    """

    # 读取 Excel 文件
    df = pd.read_excel(excel_file_path)

    # 将 DataFrame 转换为 JSON 格式，并确保输出是数组形式
    df.to_json(json_file_path, orient='records', indent=4)

    print(f"JSON file has been saved to: {json_file_path}")


'''
def resize_images(source_folder, output_folder, target_size=(224, 224)):
    """
    Resizes all images in a folder to the target size and saves them as JPEG format (ignoring transparency).

    Parameters:
    - source_folder: str, the folder path containing the original images.
    - output_folder: str, the folder path to save the resized images.
    - target_size: tuple, the target size of the images (width, height).
    Input: Original dataset images
    Output: Resized images
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        logging.error(f"An error occurs when creating the output folder: {e}")
        return

    files = os.listdir(source_folder)

    # Traverse files
    for file_name in files:
        try:
            file_path = os.path.join(source_folder, file_name)

            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(file_path) as img:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')

                    img_resized = img.resize(target_size, Image.LANCZOS)

                    save_path = os.path.join(output_folder, file_name)
                    img_resized.save(save_path, format='JPEG')
                    
        except Exception as e:
            logging.error(f"An error occurs when processing {file_name}: {e}")

'''

def resize_images(source_folder, output_folder, target_size=(224, 224)):
    """
    Resizes all images in a folder to the target size and saves them as JPEG format (ignoring transparency).

    Parameters:
    - source_folder: str, the folder path containing the original images.
    - output_folder: str, the folder path to save the resized images.
    - target_size: tuple, the target size of the images (width, height).

    Input: Original dataset images
    Output: Resized images
    """
    try:
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        logging.error(f"Error occurred while creating the output folder: {e}")
        return

    # Get all files in the source folder
    files = os.listdir(source_folder)

    # Traverse through all files in the folder
    for file_name in files:
        try:
            file_path = os.path.join(source_folder, file_name)

            # Process only image files (e.g., .png, .jpg, .jpeg)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(file_path) as img:
                    # If the image is in RGBA mode (with transparency), convert to RGB mode
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    # If the image is in P mode (palette-based), also convert to RGB mode
                    elif img.mode == 'P':
                        img = img.convert('RGB')

                    # Resize the image
                    img_resized = img.resize(target_size, Image.LANCZOS)

                    # Define the save path, ensuring the file extension is .jpg
                    save_path = os.path.join(output_folder, file_name.rsplit('.', 1)[0] + '.jpg')

                    # Save the image as JPEG format
                    img_resized.save(save_path, format='JPEG')

                    logging.info(f"Successfully resized and saved: {file_name}")

        except Exception as e:
            logging.error(f"Error occurred while processing {file_name}: {e}")

    logging.info("Processing completed.")



def rename_images_from_excel(excel_file_path, images_folder_base, target_folder, log_file_path):
    """
    This function renames images based on the mappings provided in an Excel file.

    The function reads an Excel file where:
    - The 'old_id' column contains the original image names (without extensions).
    - The 'id' column contains the new image names (which will be padded to six digits).
    - The 'style' column determines from which subfolder (Photo, Art, Cartoon, Sketch) the image is located.

    The function copies images from their respective folders (based on the 'style' column) into a target folder,
    renames them according to the 'id' column, and ensures all images are saved with a .jpg extension.

    Parameters:
    - excel_file_path (str): The path to the Excel file containing the image mapping.
    - images_folder_base (str): The base directory where the subfolders ('Photo', 'Art', 'Cartoon', 'Sketch') are located.
    - target_folder (str): The directory where the renamed images will be saved.
    - log_file_path (str): The path to the log file where the names of the images not found will be saved.
    
    Input: /F4D/anno/Annotation.xlsx; The image whose id includes a domain type letter.
    Output:The images are numbered from 000000 to 100000.
    """
    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)

    # If target folder does not exist, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Open the log file to write missing images
    with open(log_file_path, 'w') as log_file:
        # Loop through each row in the DataFrame
        for index, row in df.iterrows():
            # Get original image name and new image name (without extensions)
            original_image_name = row['old_id']  # Original image name (without extension)
            new_image_name = str(row['id']).zfill(6)  # New image name, padded to 6 digits

            # Determine the folder based on 'style' column
            folder_prefix = row['style']  # Use 'style' column to determine folder
            if folder_prefix == 'p':
                source_folder = os.path.join(images_folder_base, 'Photo')
            elif folder_prefix == 'a':
                source_folder = os.path.join(images_folder_base, 'Art')
            elif folder_prefix == 'c':
                source_folder = os.path.join(images_folder_base, 'Cartoon')
            elif folder_prefix == 's':
                source_folder = os.path.join(images_folder_base, 'Sketch')
            else:
                print(f"Unknown folder prefix: {folder_prefix}, skipping this row.")
                continue

            # Search for the image in the folder
            found_image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.png']:  # Check common image formats
                potential_path = os.path.join(source_folder, original_image_name + ext)
                if os.path.exists(potential_path):
                    found_image_path = potential_path
                    break  # Stop once the first match is found

            # If the image is found, copy and rename it
            if found_image_path:
                new_image_path = os.path.join(target_folder, new_image_name + '.jpg')  # Use .jpg extension

                try:
                    shutil.copy(found_image_path, new_image_path)  # Copy and rename the image
                    print(f"Successfully copied and renamed: {original_image_name} -> {new_image_name}.jpg")
                except Exception as e:
                    print(f"Error copying and renaming {original_image_name} to {new_image_name}.jpg: {e}")
            else:
                # If image not found, log it in the file
                log_file.write(f"{original_image_name}\n")
                print(f"Image {original_image_name} not found in folder {source_folder}, skipping this row.")

    print("Processing completed.")
def txt_to_excel(txt_file_path, excel_file_path):
    """
    Convert a .txt file to an Excel file. The .txt file is expected to have the following format:
    - The first column is the image id (e.g., '000001.jpg')
    - The subsequent columns contain the attribute values (e.g., 1 or -1 for each attribute).

    Parameters:
    - txt_file_path (str): Path to the input .txt file.
    - excel_file_path (str): Path to save the output .xlsx file.
    """
    # Read the txt file
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    # Define column names
    columns = ['id'] + [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
        "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
        "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
        "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
        "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
        "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
        "Young"
    ]

    # Process each line, split data and construct a DataFrame
    data = []
    for line in lines:
        parts = line.split()
        image_id = parts[0]
        attributes = parts[1:]
        data.append([image_id] + attributes)

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame as an Excel file
    df.to_excel(excel_file_path, index=False)
    print(f"Excel file saved at: {excel_file_path}")

import os


def rename_and_pad_zeroes_in_folder(folder_path):
    """
    Renames all files in the given folder by extracting the numeric part of each filename,
    padding it with leading zeros to ensure it is 5 digits long, and retaining the original file extension.

    Input:1.jps
    Output:00001.jpg
    """
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Process only files (not directories)
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Extract the numeric part of the filename (remove the extension)
            name_without_extension = os.path.splitext(filename)[0]

            try:
                # Try to convert the numeric part of the filename into an integer and pad it to 5 digits
                new_name = str(int(name_without_extension)).zfill(5)
                # Get the file extension
                file_extension = os.path.splitext(filename)[1]
                # Construct the new filename with the extension
                new_name_with_extension = new_name + file_extension
                # Get the full paths of the old and new filenames
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name_with_extension)
                # Rename the file
                os.rename(old_path, new_path)
                print(f'Renamed: {filename} -> {new_name_with_extension}')
            except ValueError:
                # If the filename cannot be converted to a number, raise an error
                raise ValueError(f"File '{filename}' does not appear to be a valid number.")

def add_number_to_filenames(val_folder_path, number_to_add):
    """
    Rename and Merge the Images from the FairFace 'val' and 'train' Folders
    """
    # Iterate over all files in the 'val' folder
    for filename in os.listdir(val_folder_path):
        # Process only files (not directories)
        if os.path.isfile(os.path.join(val_folder_path, filename)):
            # Extract the numeric part of the filename (without the extension)
            name_without_extension = os.path.splitext(filename)[0]
            file_extension = os.path.splitext(filename)[1]

            try:
                # Attempt to convert the filename part to an integer
                numeric_part = int(name_without_extension)

                # Add the specified number to the numeric part of the filename
                new_numeric_part = numeric_part + number_to_add

                # Construct the new filename by converting the new numeric part to a string
                new_filename = str(new_numeric_part) + file_extension

                # Get the full paths of the old and new filenames
                old_path = os.path.join(val_folder_path, filename)
                new_path = os.path.join(val_folder_path, new_filename)

                # Rename the file
                os.rename(old_path, new_path)
                print(f'Renamed: {filename} -> {new_filename}')

            except ValueError:
                # Raise an error if the filename cannot be converted to a valid integer
                raise ValueError(f"File '{filename}' cannot be converted to a valid integer, cannot rename.")

def main():
    # Need further adjustment...
    '''
    #resize_images()
    source_folder1 = '/home/lym/FairDistBench/datasets/CelebA/raw/img_align_celeba'
    output_folder1 = '/home/lym/FairDistBench/datasets/CelebA/resized'
    print("开始处理CelebA图片...")
    resize_images(source_folder1, output_folder1)

    source_folder2 = '/home/lym/FairDistBench/datasets/UTKFace/raw/UTKFace'
    output_folder2 = '/home/lym/FairDistBench/datasets/UTKFace/resized'
    print("开始处理UTKFace图片...")
    resize_images(source_folder2, output_folder2)

    source_folder3 = '/home/lym/FairDistBench/datasets/F4D/raw'
    output_folder3 = '/home/lym/FairDistBench/datasets/F4D/resized'
    print("开始处理F4D图片...")
    resize_images(source_folder3, output_folder3)

    source_folder4 = '/home/lym/FairDistBench/datasets/FairFace/raw/FairFace/train'
    output_folder4 = '/home/lym/FairDistBench/datasets/FairFace/resized'
    print("开始处理FairFace图片...")
    resize_images(source_folder4, output_folder4)
    '''
    '''
    # rename_images_from_excel()
    excel_file_path = '/home/lym/FairDistBench/datasets/F4D/anno/Annotation.xlsx'
    images_folder_base = '/home/lym/MBDG/FairPACS'
    target_folder = '/home/lym/FairDistBench/datasets/F4D/raw'
    log_file_path = '/home/lym/FairDistBench/datasets/F4D/anno/not_found.txt'
    rename_images_from_excel(excel_file_path, images_folder_base, target_folder, log_file_path)
    '''
    #'''
    #excel_to_json()
    excel_file_path = '/home/lym/FairDistBench/datasets/F4D/anno/Annotation.xlsx'
    json_file_path = '/home/lym/FairDistBench/datasets/F4D/anno/f4d.json'
    excel_to_json(excel_file_path, json_file_path)
    excel_file_path1 = '/home/lym/FairDistBench/datasets/FairFace/anno/fairface.xlsx'
    json_file_path1 = '/home/lym/FairDistBench/datasets/FairFace/anno/fairface.json'
    excel_to_json(excel_file_path1, json_file_path1)
    #'''
    '''
    #txt_to_excel()
    txt_file_path = '/home/lym/FairDistBench/datasets/CelebA/anno/list_attr_celeba.txt'
    excel_file_path = '/home/lym/FairDistBench/datasets/CelebA/anno/celeba.xlsx'
    txt_to_excel(txt_file_path, excel_file_path)
    
    #rename_and_pad_zeroes_in_folder()
    rename_and_pad_zeroes_in_folder('/home/lym/FairDistBench/datasets/FairFace/raw/FairFace/train')
    '''

if __name__ == '__main__':
    main()