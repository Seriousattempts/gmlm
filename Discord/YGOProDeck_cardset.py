import requests
import pandas as pd
import time

# Make a request to the endpoint
response = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php')

# Parse the JSON response
data = response.json()

# Initialize an empty list to hold the card data
cards_data = []

# Loop through each card
for card in data['data']:
    # Check if 'card_sets' is in the card dictionary
    if 'card_sets' in card:
        # Loop through each set in the card's card_sets field
        for set in card['card_sets']:
            # Create a dictionary with the card's id and the set's set_code
            card_data = {'id': card['id'], 'set_code': set['set_code']}
            # Add the dictionary to the list of card data
            cards_data.append(card_data)
            print(f"Added card with id {card['id']} and set code {set['set_code']} to list.")

            # If we've added 20 cards, pause for a second
            if len(cards_data) % 20 == 0:
                print("Added 20 cards, pausing for a second...")
                time.sleep(1)

# Convert the list of card data to a DataFrame
print("Converting data to DataFrame...")
df = pd.DataFrame(cards_data)

# Write the DataFrame to an Excel file
print("Writing DataFrame to Excel file...")
df.to_excel('C:\\Users\\C\\D\\ART\\EXAMPLES\\YGO\\PLOT\\cards_data.xlsx', index=False)
print("Done!")