from PIL import Image
img = Image.open('icon.jpg')
img.save('app.ico', sizes=[(256, 256)])