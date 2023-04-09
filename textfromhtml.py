import os
from bs4 import BeautifulSoup
from embedd import EmbeddingDataFrame
input_folder = "scraped_pages"
output_folder = "extracted_text"

def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    text = soup.get_text()
    
    # Remove leading and trailing whitespace and collapse consecutive whitespace characters
    text = ' '.join(text.strip().split())
    
    return text

embeddedDF = EmbeddingDataFrame()

embeddedDF.generate_df()

def process_files():
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as input_file:
                    html = input_file.read()
                    text = extract_text_from_html(html)
                    url = file_path.replace(".html", "").replace("scraped_pages/","https://")
                    embeddedDF.add_chunks_to_df(url, text)
    embeddedDF.save_df("embeddedDF.pickle")
if __name__ == "__main__":
    process_files()