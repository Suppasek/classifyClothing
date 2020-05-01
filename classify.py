# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2 
from PIL import Image

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# plt.imshow(img)
	plt.show()
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0


	return img

def resize(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    background.convert('RGB')
    background.save('./img/resize.png')

def crop_image(fileName):
	img = cv2.imread(fileName,0)

	ret,thresh = cv2.threshold(img,255,255,255)
	contours,hierarchy = cv2.findContours(thresh, 1, 2)

	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

	biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

	cnt = biggest_contour
	
	x,y,w,h = cv2.boundingRect(cnt)
	print(x)
	print(y)
	print(w)
	print(h)
	#draw contours eiei
	#img2 = cv2.rectangle(img,(x,y),(x+w,y+h),(200,200,255),4)

	#Crop image !!!!!!!!!!!!!!!!!!!!!!!!!!!
	crop_img = img[y:y+h, x:x+w]

	#write img to this machine
	cv2.imwrite('./img/cropped.png',crop_img)

	#show image
	# cv2.imshow('img',crop_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

def remove_skin(fileName):
	img = cv2.imread(fileName)
	min_YCrCb = np.array([0,133,77],np.uint8)
	max_YCrCb = np.array([235,173,127],np.uint8)
	converted = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	skinMask = cv2.inRange(converted,min_YCrCb,max_YCrCb)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	mask = cv2.bitwise_not(skinMask)
	remove_skin = cv2.bitwise_and(img, img, mask=mask)

	cv2.imwrite('./img/remove_skin.png',remove_skin)


# load an image and predict the class
def run_example(fileName):

	#remove_skin(fileName)
	crop_image(fileName)
	im = Image.open('./img/cropped.png')
	
	resize(im, 28, 28)

	img = load_image('./img/resize.png')

	# load model
	model = load_model('final_model.h5')
	# predict the class
	result = model.predict(img)
	predict = model.predict_classes(img)
	# result = model.predict_classes(img)
	# print(result)
	# print(class_names[np.argmax(result[0])])
	print(class_names[predict[0]])

# entry point, run the example
run_example('./img/segment_img.jpg')