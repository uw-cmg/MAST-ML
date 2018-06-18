import xml.etree.ElementTree as ET
import time
import os

def make_html(image_paths, savepath):
    #begining
    html = ET.Element('html')
    title = ET.SubElement(html, 'title')
    title.text = 'MASTML'
    body = ET.SubElement(html, 'body')
    h1 = ET.SubElement(body, 'h1')
    h1.text = 'MAST Machine Learning Output'
    par = ET.SubElement(body, 'p')
    par.text = str(time.time())
    ET.SubElement(body, 'hr')

    for path in image_paths:
        a = ET.SubElement(body, 'a')
        a.set('href', path)
        img = ET.SubElement(a, 'img')
        img.set('src', path)

    # ending
    ET.SubElement(body, 'hr')
    h2 = ET.SubElement(body, 'h2')
    h2.text = 'Setup'

    #logpath = os.path.join(savepath)
    #ET.SubElement(self.body, 'a', {'href':logpath}).text = 'Log File'
    #ET.SubElement(self.body, 'br')

    #confpath = os.path.join(self.save_path, str(self.configfile))
    #confpath = os.path.relpath(confpath, self.save_path)
    #ET.SubElement(self.body, 'a', {'href': confpath}).text = 'Config File'
    ##ET.SubElement(self.body, 'br')
    #ET.SubElement(self.body, 'hr')

    with open(os.path.join(savepath, 'index.html'), 'w') as f:
        f.write(ET.tostring(html, encoding='unicode', method='html'))
