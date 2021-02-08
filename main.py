import cv2
import matplotlib.pyplot as plt
from IPython.display import Image,display_jpeg
from sklearn.externals import joblib

def predict_digit(filename):
  # 学習済みモデルを読み込む
  clf = joblib.load("digits.pkl")
  # 自分で用意した手書きの画像ファイルを読み込む
  img = cv2.imread(filename)
  # 画像データを学習済みデータに合わせる
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (8, 8))
  img = 15 - img
  plt.imshow(img, cmap="gray")
  plt.show()
  img = img.reshape((-1, 64))
  # データを予測する
  res = clf.predict(img)
  return res[0]

# 画像ファイルを指定して実行
n = predict_digit("3_1.png")
print("" + " = " + str(n))