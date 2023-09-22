import json

def json_viewer(json_file_path):

    try:
        # Open and read the JSON file
        with open(json_file_path, 'r') as json_file:
            # Parse the JSON data into a Python data structure (typically a dictionary or list)
            data = json.load(json_file)

            # Check if the JSON data is a list and has at least one element
            if isinstance(data, list) and len(data) > 0:
                # Print the first entry (object) in the list
                print("First entry in the JSON file:")
                print(data[0])
            elif isinstance(data, dict):
                # If the JSON data is a dictionary, print the whole dictionary
                print("The JSON data is a dictionary:")
                print(data)
            else:
                print("The JSON data is not a list or dictionary, unable to process.")
            
    except FileNotFoundError:
        print(f"The file '{json_file_path}' does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
