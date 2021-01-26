# Creating GUI with tkinter
# https://morioh.com/p/b24bf58a2e2e

import tkinter
from tkinter import *
from chatbot import chatbot_response


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "Ty: " + msg + '\n\n')
        ChatLog.config(foreground="#474545", font=("Verdana", 11))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED) # NORMAL - można pisać
        ChatLog.yview(END)


base = Tk()
base.title("ChatBot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", fg="#474545", height="8", width="50", font="Verdana", padx=8, pady=8)

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="arrow")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Wyślij", width="12", height=5,
                    bd=0, bg="#4FA9DA", activebackground="#FFFFFF", fg='#ffffff',
                    command=send, anchor=CENTER)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", fg="#474545", width="29", height="5", font=("Verdana", 11), padx=8, pady=8)

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=271, y=401, height=90, width=105)

base.mainloop()
