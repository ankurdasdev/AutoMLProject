import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            texts = []
            for node in tree.iter():
                if node.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t':
                    if node.text:
                        texts.append(node.text)
                elif node.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p':
                    texts.append('\n')
            
            return ''.join(texts).replace('\n\n', '\n')
    except Exception as e:
        return str(e)

import sys
sys.stdout.reconfigure(encoding='utf-8')
print(extract_text_from_docx(r'd:\IOT Project\IOT Salesforce Project Design Document (1).docx'))
