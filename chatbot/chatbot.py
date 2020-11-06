from tkinter import *
from model import Model
from random import randint


class Win(Frame):

    MOCK_MODEL = False
    GREETING_MESSAGE = "Hello, and welcome!\n"

    def __init__(self, master=None, blob_file=None):
        Frame.__init__(self, master)
        self.master = master
        self.master.resizable(width=False, height=False)
        self.scroller, self.chat_box, self.response_box, self.submit_button = None, None, None, None
        self.last_sentence = ""
        self.init_win()
        self.blob = ['It is true but not in the north.']
        if blob_file is not None:
            try:
                with open(blob_file, 'r') as fh:
                    data = fh.read()
                    self.blob += data.split('\n')
            finally:
                pass

        self.model = None
        self.init_model()

    def init_model(self):
        self.model = Model()

    def init_win(self):
        self.master.title("~ chatbot ~")
        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)
        self.scroller = Scrollbar(self)
        self.chat_box = Text(self, height=32, width=47, borderwidth="0")
        self.response_box = Text(self, height=3, width=36, borderwidth="0")
        # creating a button instance
        self.submit_button = Button(self, text="SEND",
                               command=self.send_msg, bg="lightblue",
                               borderwidth="0", highlightcolor="black",
                               height="3", width="10")

        self.scroller.pack(side=RIGHT, fill=Y)
        self.scroller.config(command=self.chat_box.yview)
        self.chat_box.place(x=1, y=1)
        self.response_box.place(x=1, y=540)
        self.submit_button.place(x=300, y=540)

        # configs
        self.chat_box.tag_configure('bot',
                                    foreground='#476042',
                                    font=('Tempus Sans ITC', 12, 'bold'))
        self.chat_box.tag_configure('user',
                                    foreground='#3953FA',
                                    font=('Tempus Sans ITC', 12, 'bold'))

        self.response_box.bind("<Key>", self.enter_key)
        self.chat_box.insert(END, "Bot >> {}".format(self.GREETING_MESSAGE), 'bot')
        self.response_box.focus_set()

    def enter_key(self, event):
        if event.char == '\r':
            self.send_msg()

    def send_msg(self):
        # add msg to main bot window
        self.last_sentence = self.response_box.get("1.0", END).strip()
        self.response_box.delete("1.0", END)
        self.response_box.delete("1.0")
        if self.last_sentence != "":
            self.chat_box.insert(END, "User>> {}\n".format(self.last_sentence), 'user')
            self.call_responder(randone=self.MOCK_MODEL)
            self.chat_box.see(END)

    def update_response(self, response):
        self.chat_box.insert(END, "Bot >> {}\n".format(response.strip()), 'bot')

    def call_responder(self, randone=False):
        if randone:
            resp = self.blob[randint(0, len(self.blob)-1)]
        else:
            #resp = self.model.predict(self.last_sentence)
            resp = self.model.print_result(self.last_sentence)
            #resp = api-call to model with "self.last_sentence" return string
        self.update_response(resp)


root = Tk()
root.geometry("400x600")
app = Win(root, blob_file='blob')
root.mainloop()
