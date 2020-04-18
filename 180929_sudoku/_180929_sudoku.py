

import time             #時間を扱うモジュール
import pyocr            #画像データから文字を抽出するモジュール
from PIL import Image   #画像処理ライブラリ
import pyocr.builders  
import pyocr
import cv2
import numpy as np
import sys
#from keras.models import load_model
from tkinter import messagebox
import os
import pprint
from matplotlib import pyplot as plt

#◆◆【関数】画像を表示　◆◆
def image_show(image):

    print('********** 画像を表示 **********')
    cv2.imshow('1',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#◆◆　画像を整形する(角度のついた画像を正方形へ変換　◆◆
def shaping_images(img):
    if img is none:
        print('ファイルを込めません(shaping_image)')
        sys.exit()

    rows.cols = img.shape[:2]

#◆◆　枠を描画する　◆◆
def select_view(img,contours):

    print('********** 画像内の表示範囲を出力、保存 **********')
    #for i in contours:
     #   x,y,w,h = cv2.boundingRect(i)
      #  cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    x,y,w,h = cv2.boundingRect(contours[0])
    img = img[y:y+h,x:x+w]
    image_show(img)
    #cv2.imwrite('C:/Users/amur_/Downloads/open cv save folder/image.jpg',img)

#◆◆　OCRで数字を認識する。◆◆
def get_digital_ocr_info(img):

    print('********** OCRで数字を認識 **********')

    TESSERACT_PATH = 'C:/Program Files (x86)/Tesseract-OCR'
    TESSDATA_PATH = 'C:/Program Files (x86)/Tesseract-OCR/tessdata'

    os.environ["PATH"] += os.pathsep + TESSERACT_PATH
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH

    result = None
    start_time = time.time()
    print("********** start convert_image_to_deadline **********")

    #width, height =  img.size

    tools = pyocr.get_available_tools ()      #OCRツールの有無を確認
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    tool = tools[0]
    print(tool)
    print("will use tool '%s'" % (tool.get_name())) #使用するOCRツールの名前が出ます。変えたい場合は1個前の参照先を変えること。
    
    langs = tool.get_available_languages()  #使用するできる言語の確認
    print("Available languages: %s" % ", ".join(langs))

    lang = langs[0]
    print("will use lang '%s'" %(lang)) #使用する言語について

    digit_txt = tool.image_to_string(
        Image.open(img),
        lang=lang,
        builder = pyocr.builders.DigitBuilder(tesseract_layout=6)
    )

    print('DigitBuilder',digit_txt)

    print('********** end convert_image_to deadline **********')

    return digit_txt

#◆◆　画像を整形　◆◆
def shaping_image(img):
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    area_chk = 0
    image_y,image_x = image.shape
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])

        if (h) >= (image_y*0.98) or (w) >= (image_x*0.98):
            continue
        elif (h) < ((w)*0.7) or (w) < ((h-y)*0.7):
            continue
        elif (area_chk)<((w)*(h)):
            area_chk = (w)*(h)
            mx=x
            my=y
            mw=w
            mh=h
            
    image = img[my:my+mh,mx:mx+mw]

    #マスを整形
    image, contours, hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    mw,mh = 0,0
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        if mw*mh < w*h :
            mw,mh = w,h
            contours_max=contours[i]

    w,h,x,y = cv2.boundingRect(contours_max)

    epsilon = 0.01 * cv2.arcLength(contours_max,True)
    approx = cv2.approxPolyDP(contours_max,epsilon,True)
    #print(approx)
    #if len(approx) == 4:
        #cv2.drawContours(image, [approx], -1, (255, 0, 0), 3)
    a,b,c,d=[approx[0][0][0],approx[0][0][1]],[approx[1][0][0],approx[1][0][1]],[approx[2][0][0],approx[2][0][1]],[approx[3][0][0],approx[3][0][1]]

    if a[0] > x*0.5:
        list_src = np.float32([b,c,d,a])
    else:
        list_src = np.float32 ([a,b,c,d])
    if x > y:
        list_dst = np.float32([[0,0],[0,y],[y,y],[y,0]])
    elif x <= y:
        list_dst = np.float32([[0,0],[0,x],[x,x],[x,0]])
    perspective_matrix = cv2.getPerspectiveTransform(list_src,list_dst)
    dst = cv2.warpPerspective(image,perspective_matrix,(x,y))
    
    if x > y :
        fit_mas = dst[0:y,0:y]
        mh = y 
        mw = y
    elif y > x :
        fit_mas = dst[0:x,0:x]
        mw = x
        mh = x

    return fit_mas,mw,mh

#◆◆　画像を認識しやすく処理し、画像の数字を認識させる　◆◆
def start_sudoku(adress):

    print('********** メインコード **********')
    #画像を読み込み　->　グレイスケールへ変換
    gray = cv2.imread(adress,cv2.IMREAD_GRAYSCALE)
    #二値化処理
    res,bw = cv2.threshold(gray,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #ネガポジ反転
    nega = cv2.bitwise_not(bw)

    #撮影画像かネット収取画像かを仕分け
    image, contours, hierarchy = cv2.findContours(nega, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1 :
        fit_mas,mw,mh = shaping_image(nega)
    elif len(contours) == 1 :
        x,y,w,h = cv2.boundingRect(contours[0])
        fit_mas = nega[y:y+h,x:x+w]
        mw,mh = fit_mas.shape

    #解読用配列を用意
    number = [[0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0]]

    #整形したマスより数字を抽出
    for b in range(9):
        for a in range(9):
            #ざっくり数字部分を切り取り
            cut = fit_mas[b*mh//9 : (b+1)*mh//9 , a*mw//9 : (a+1)*mw//9]
            #ざっくりから数字のみを抽出
            image, contours, hierarchy = cv2.findContours(cut, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                x,y,w,h = cv2.boundingRect(contours[i])
                #不要物排除
                if w > mw//9*0.95 or h > mh//9*0.95 or w*h < (mw//9 * mh//9 * 0.13) : #x < a//9*0.1 or y < b//9*0.1 or x > w or y > h
                    number[b][a]=0
                    continue

                #文字サイズ縮小し、周りを黒塗り
                #a,b = cut[y:y+h,x:x+w].shape
                brk = np.zeros((1,1),np.uint8)
                #画像全体の高さを変数へ a:幅、b:高さ
                wi=round(h*1.7)
                hi=round(h*1.9)
                brk_img = cv2.resize(brk,(wi,hi))
                brk_img[hi//2-h//2 : hi//2+(h-h//2), wi//2-w//2 : wi//2+(w-w//2)] = cut[y:y+h,x:x+w]

                #OCRに認識させるため、jpg形式で保存
                save_place = 'C:/Users/amur_/Downloads/open cv save folder/number.jpg'
                cv2.imwrite(save_place,brk_img)

                #画像から数時を確認
                number[b][a] = int(get_digital_ocr_info(save_place))

                #image_show(img_cng)
                break

    return number,fit_mas

#◆◆　画像をUSERに選択させる　◆◆
def picture_input():
    # モジュールのインポート
    import os,tkinter,tkinter.filedialog,tkinter.messagebox

    #ファイルの選択
    root = tkinter.Tk()
    root.withdraw()
    ftyp = [("","*")]
    idir = os.path.abspath(os.path.dirname(__file__))
    tkinter.messagebox.showinfo("〇×プログラム","処理ファイルを選択してください。")
    file = tkinter.filedialog.askopenfilename(filetypes = ftyp,initialdir = idir)

    return file

#◆◆　数読を解読　◆◆
#縦、横、マス内、の数字の重複確認
def chksu(suary,a,b,su):
    for j2 in range(9):
        if j2 != a:
            if suary[b][j2] == su:
                return False
            
    for j1 in range(9):
        if j1 != b:
            if suary[j1][a] == su:
                return False

    j1s = (int((a+3)/3)-1)*3
    j2s = (int((b+3)/3)-1)*3

    for g2 in range(j2s,j2s+3):
        for g1 in range(j1s,j1s+3):
            if suary[g2][g1] == su:
                return False
    return True

#空白マスを確認
def getblank(suary,a,b):
    for b in range(9):
        for a in range(9):
            if suary[b][a]==0:
                return True,a,b
    return False,a,b

#空白マスに数字を追加
def trysu(suary):

    a=0
    b=0
    su=0

    blank,a,b=getblank(suary,a,b)

    if blank==False:
        return True
 
    for su in range(1,10):
        if chksu(suary,a,b,su) == True:
            suary[b][a] = su
            if trysu(suary) ==True:
                return True

    suary[b][a]=0

#数読解読メイン
def main(suary):
    
    a=0
    b=0

    resurt=trysu(suary)

    blank,a,b=getblank(suary,a,b)
    if blank == False:
        messagebox.showinfo('ナンプレ解読','解読成功！！')

    else:
        messagebox.showinfo('ナンプレ解読','解読失敗！！')

    return suary

def result_sudok(number,result,image):

    image = cv2.bitwise_not(image)
    
    adress = 'C:/Users/amur_/Dropbox/09_programing/repos/180929_sudoku_/number/'
    for b in range(9):
        for a in range(9):
            if number[b][a] == 0:
                picture = adress + str(result[b][a]) + '.jpg'
                picture = cv2.imread(picture,cv2.IMREAD_GRAYSCALE)
                x,y = image.shape
                resize_picture = cv2.resize(picture,(x//11,y//11))
                w,h = resize_picture.shape
                row =    (y//9)//2+b*(y//9)+y//86
                column = (x//9)//2+a*(x//9)+x//86
                image[row-(h//2) : row+(h-h//2) , column-(w//2) : column+(w-w//2)] = resize_picture[0:h,0:w]

                #image_show(image)

    return image

#◆◆　スタート　◆◆
if __name__ == '__main__':
    print('********** スタート **********')
    #adress = 'C:/Users/amur_/Dropbox/09_programing/01_sudok/camera_picture/P_20190127_104238.jpg'

    adress = picture_input()

    number,image = start_sudoku(adress)

    result = [[0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0]]

    for b in range(9):
        for a in range(9):
            result[b][a] = number[b][a]

    result = main(result)

    image = result_sudok(number,result,image)

    image_show(image)

    save = adress.replace('.jpg','result.jpg')

    cv2.imwrite(save,image)
