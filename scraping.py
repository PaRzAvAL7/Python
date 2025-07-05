import requests
from bs4 import BeautifulSoup

# Step 1: Choose a URL to scrape
url = "https://quotes.toscrape.com"

# Step 2: Send a request to the website
response = requests.get(url)

# Step 3: Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Step 4: Extract the data
quotes = soup.find_all("div", class_="quote")

# Step 5: Loop through and print structured data
for quote in quotes:
    text = quote.find("span", class_="text").get_text()
    author = quote.find("small", class_="author").get_text()
    tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]

    print("Quote:", text)
    print("Author:", author)
    print("Tags:", tags)
    print("-" * 50)
