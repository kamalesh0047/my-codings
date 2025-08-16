import matplotlib.pyplot as plt
from mtcnn import MTCNN
from google.colab import files
uploaded=files.upload()
for fn in uploaded.keys():
  img_path=fn
img=plt.imread(img_path)
detector=MTCNN()
faces=detector.detect_faces(img)
plt.imshow(img)
ax=plt.gca()
for face in faces:
  x,y,w,h=face['box']
  rect=plt.Rectangle((x,y),w,h,fill=False,color='red')
  ax.add_patch(rect)
plt.show()
