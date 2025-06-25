import tkinter as tk
from tkinter import Tk
from control import model


def format_currency(num):
    return f"{num:,}".replace(",", ".")

def predict_sell_price():
    if((not input_users[0]) or (not input_users[1]) or (not input_users[2])):
        pass

    x_variables =  [float(input_users[0].get()) *1000, float(input_users[1].get())*1000, float(input_users[2].get()), float(input_users[3].get())]
    predict_price = model.predict(x_variables)
    other_labels.config(text=format_currency(int(predict_price)), foreground='Green')


mainWindow = tk.Tk()
mainWindow.title("SMARTPHONE PRICE PREDICT TOOLS")
mainWindow.geometry('360x240')


Labels = ['Memory (GB):', 'Storage (GB):', 'Original Price: Rp.', 'Discount: Rp.', 'Selling Price: Rp.']

frames = [tk.Frame(mainWindow, width=350, height=30) for i in range(len(Labels))]
label_frame = [tk.Label(frames[i], text= Labels[i] ,font=('helvetica', 10, 'bold')) for i in range(len(Labels))]
label_frame[-1].config(foreground='Green')

input_users = [tk.Entry(frames[i], text="", font=('helvetica', 12, 'bold'), border=1, relief='solid') for i in range(len(Labels))]
other_labels = tk.Label(frames[4], text="", font=('helvetica', 12, 'bold'), border=1, relief='solid')
other_labels.place(x=125, y=2.25, width=220)

predict_butt = tk.Button(mainWindow, text="Predict", font=('helvetica', 12, 'bold'), command=predict_sell_price)
predict_butt.place(x=130, y=162.5)


for i in range(len(Labels)):
    frames[i].place(x=5, y=(30 * i) + 2.5)
    label_frame[i].place(x=0, y=3)

    if(i < len(Labels) - 1):
        input_users[i].place(x=125, y=2.25, width=220)

mainWindow.mainloop()
