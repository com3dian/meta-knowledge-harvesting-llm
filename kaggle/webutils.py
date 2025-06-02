import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

def readWebContent(url: str) -> BeautifulSoup:
    """
    Reads the content of a webpage and returns a BeautifulSoup object.
    :param url: The URL of the webpage to read.
    :return: A BeautifulSoup object containing the parsed HTML content.
    """
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    else:
        print("Failed to retrieve the webpage.")
        return None

def downloadAndParseXML(url):
    """
    Downloads metadata XML from the Wadden data service and parses it.
    Params: url (str): The URL of the XML to download.
    Returns: tuple: (xml_str, root) where xml_str is the raw XML text and root is the parsed ElementTree root
    """

    # Download the XML
    response = requests.get(url)
    xml_str = response.text
    
    # Parse the XML
    root = ET.fromstring(xml_str)
    
    return xml_str, root