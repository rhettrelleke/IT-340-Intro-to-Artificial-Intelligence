# Step 1: Read and process the data to create a nested dictionary

# Initialize an empty dictionary to store the scholar information.
scholars_data = {}

# Read "Experts.txt" to gather information about scholars in computer science.
with open("Experts.txt", "r", encoding="utf-8") as experts_file:
    for line in experts_file:
        fields = line.strip().split('\t')
        if len(fields) == 4 and "Computer Science" in fields[3]:
            scholar_id, university, name, _ = fields
            # Initialize the scholar's university if not already present in the dictionary.
            if university not in scholars_data:
                scholars_data[university] = {}
            # Add scholar's name and ID to the university's dictionary.
            scholars_data[university][name] = {"ID": scholar_id}

# Read "Profiles.txt" to gather information about the areas studied by scholars.
with open("Profiles.txt", "r", encoding="utf-8") as profiles_file:
    for line in profiles_file:
        fields = line.strip().split('\t')
        scholar_id, areas_studied = fields[0], fields[1]
        # Update the areas studied for each scholar in the dictionary.
        for university, scholars in scholars_data.items():
            for name, info in scholars.items():
                if info["ID"] == scholar_id:
                    info["Areas"] = areas_studied.split(';')

# Step 2: Implement a search function for partial matches

def search_scholars(input_string):
    best_matches = []
    max_word_count = 0

    # Split the input string into individual words for matching.
    input_keywords = input_string.lower().split()

    # Iterate through scholars and their areas of study to find the best matches.
    for university, scholars in scholars_data.items():
        for name, info in scholars.items():
            areas = info.get("Areas", [])
            for area in areas:
                area_keywords = area.lower().split()
                # Count the number of matching words between input and area.
                match_count = sum(keyword in area_keywords for keyword in input_keywords)
                if match_count > max_word_count:
                    max_word_count = match_count
                    best_matches = [(name, area)]
                elif match_count == max_word_count:
                    best_matches.append((name, area))

    return best_matches

# Step 3: Display search results

while True:
    search_query = input("Enter keywords to search for scholars in computer science (or 'exit' to quit): ")
    if search_query.lower() == "exit":
        break
    results = search_scholars(search_query)
    
    if results:
        print("Search Results:")
        for name, area in results:
            print(f"{name} ({area})")
    else:
        print("No matching scholars found.")

