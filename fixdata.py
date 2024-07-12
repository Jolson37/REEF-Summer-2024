import os

def replace_sequence_in_files(folder_path, old_sequence, new_sequence):
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the current file is a text file
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, 'r') as file:
                file_content = file.read()

            # Replace the old sequence with the new sequence
            new_content = file_content.replace(old_sequence, new_sequence)

            with open(file_path, 'w') as file:
                file.write(new_content)

            print(f"Processed file: {filename}")

# Specify the folder path containing the text files
folder_path = 'datasets/Bullet_holes/test/label_update'

# Specify the sequences to be replaced
old_sequence = "0 "
new_sequence = "5 "

# Call the function to perform the replacements
replace_sequence_in_files(folder_path, old_sequence, new_sequence)
