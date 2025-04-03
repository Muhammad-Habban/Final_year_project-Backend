# Python script to remove lines ending with "/work" from a file

def remove_lines_with_work(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filter out lines that end with '/work'
    filtered_lines = [line for line in lines if not line.strip().endswith('/work')]

    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)

# Replace 'req.txt' with the path to your requirements file
file_path = 'req.txt'
remove_lines_with_work(file_path)