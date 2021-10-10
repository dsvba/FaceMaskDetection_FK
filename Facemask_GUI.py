from tkinter import *
# Tkinter 模块(Tk 接口)是 Python 的标准 Tk GUI 工具包的接口 .Python3.x 版本使用的库名为 tkinter,即首写字母 T 为小写。
from tkinter import filedialog
from PIL import Image, ImageTk

"""
创建一个GUI程序

1、导入 Tkinter 模块
2、创建控件
3、指定这个控件的 master， 即这个控件属于哪一个
4、告诉 GM(geometry manager) 有一个控件产生了。
"""

root = Tk()  # 创建窗口对象的背景色
root.title('口罩识别')
root.geometry('1280x1200')  # 定义窗口弹出时的默认展示位置
root.resizable(0, 0)
f1 = Frame(root)  # 框架控件；在屏幕上显示一个矩形区域
f1.pack()  # 包装；
f2 = Frame(root)
f2.pack()
f3 = Frame(root)
f3.pack()
f4 = Frame(root)
f4.pack()


# 调整大小
def resize_w(r_w, w, h):
    ratio = r_w / w
    w_re = int(w * ratio)
    h_re = int(h * ratio)
    return (w_re, h_re)


def resize_h(r_h, w, h):
    ratio = r_h / h
    w_re = int(w * ratio)
    h_re = int(h * ratio)
    return (w_re, h_re)


# 标签图片1
img = Image.open('BGP.png')
s = img.size
re_s = img.resize(resize_w(1280, s[0], s[1]))  # 重置大小
photo = ImageTk.PhotoImage(re_s)
thelabel = Label(f1, image=photo)
thelabel.pack()


# 选择图片
def file_select():
    global img_path, photo1
    img_path = filedialog.askopenfilename()
    img = Image.open(img_path)
    s = img.size
    re_s = img.resize(resize_h(500, s[0], s[1]))  # 重置大小
    photo1 = ImageTk.PhotoImage(re_s)
    t1.configure(image=photo1)
    t2.delete(1.0, 'end')
    t2.insert(1.0, img_path, 'tag_1')


b1 = Button(f2, text='图片选择', font=15, background='lightblue', fg='black', command=file_select)
b1.grid(row=0, column=0)  # 网格
t1 = Label(f2)
t1.grid(row=1, column=1)
t2 = Text(f2, width=80, height=1, font=('Fixdays', 15), background='lightblue', borderwidth=2, relief='flat')
t2.grid(row=0, column=1, )
t2.tag_config("tag_1", justify='center')

import cv2
import pytorch_infer


def img_recognition():
    # 读取图片
    img = cv2.imread(img_path)
    # 将图像转化为灰度图像，减少运算量
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pytorch_infer.inference(img, show_result=True, target_shape=(360, 360))


b2 = Button(f2, text='口罩识别', font=15, background='lightblue', fg='black', command=img_recognition)
b2.grid(row=1, column=0)


def Husin_facemask():
    pytorch_infer.run_on_video(0, '', conf_thresh=0.5)


b3 = Button(f2, text='口罩识别实时', font=15, background='lightblue', fg='black', command=Husin_facemask)
b3.grid(row=2, column=0)
root.mainloop()
