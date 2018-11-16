import idf
import invertedindex as invi
import pickle
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import get_results as gr
import evaluate as e

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk



option=0
wnl = WordNetLemmatizer()

class ButtonWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Button Demo")
        self.set_border_width(10)
        self.arx = []

        vbox = Gtk.Box(spacing=4, orientation = Gtk.Orientation.VERTICAL)
        self.add(vbox)

        hbox_top = Gtk.Box(spacing = 10)
        vbox_mid = Gtk.Box(spacing = 10)
        hbox_bottom = Gtk.Box(spacing = 4)
        vbox.pack_start(hbox_top, False, True, 0)
        vbox.pack_start(vbox_mid, True, True, 0)
        vbox.pack_start(hbox_bottom, False, True, 0)

        self.entry = Gtk.Entry()
        self.entry.set_size_request(250, 30)
        self.entry.set_placeholder_text("Enter the search query here: ")

        button1 = Gtk.Button.new_with_label("Search")
        button1.connect("clicked", self.on_click_me_clicked)

        button2 = Gtk.Button.new_with_mnemonic("Generate Dict")
        button2.connect("clicked", self.on_open_clicked)

        button3 = Gtk.Button.new_with_mnemonic("_Close")
        button3.connect("clicked", self.on_close_clicked)

        button4 = Gtk.Button.new_with_mnemonic("Evaluate Results")
        button4.connect("clicked", self.on_click_evaluation)

        self.check1 = Gtk.CheckButton("Embedding Search")
        self.check2 = Gtk.CheckButton("Bloom Filter")

        self.vbox_left = Gtk.Box(orientation = Gtk.Orientation.VERTICAL)
        self.vbox_mid_mid = Gtk.Box(spacing = 15, orientation = Gtk.Orientation.VERTICAL)
        self.vbox_mid_mid.set_size_request(500,500)
        self.vbox_right = Gtk.Box(orientation = Gtk.Orientation.VERTICAL)

        vbox_mid.pack_start(self.vbox_left, True, True, 0)
        vbox_mid.pack_start(self.vbox_mid_mid, False, True, 0)
        vbox_mid.pack_start(self.vbox_right, True, True, 0)

        self.arx = []
        for x in range(10):
            self.arx.append(TitleButton("Result " + str(x+1)))
            self.vbox_mid_mid.pack_start(self.arx[x], True, True, 0)
        self.vbox_left.pack_start(Gtk.Label("\n\n"), True, True, 0)
        self.vbox_right.pack_start(Gtk.Label("\n\n"), True, True, 0)

        hbox_top.pack_start(self.entry, False, True, 0)
        hbox_top.pack_end(self.check2, False, True, 0)
        hbox_top.pack_end(self.check1, False, True, 0)
        hbox_top.pack_end(button1, False, True, 0)

        hbox_bottom.pack_end(button3, False, True, 0)
        hbox_bottom.pack_end(button2, False, True, 0)
        hbox_bottom.pack_end(button4, False, True, 0)


    def on_click_evaluation(self, button):
        stringy = "This may take upto 5 minutes.\n Press OK to continue"
        x = DialogExample(self,stringy)
        x.run()
        x.destroy()
        stringy = "Evaluation for normal search\n" + e.normal_evaluation()
        stringy += "\nEvaluation for Embedding search\n" + e.embedding_evaluation(model)
        x = DialogExample(self,stringy)
        x.run()
        x.destroy()

    def on_click_me_clicked(self, button):
        global model
        for doc in range(10):
            self.arx[doc].set_label("Result " + str(doc+1))

        if self.check1.get_active():
            query=self.entry.get_text()
            added_vocab = gr.query_expansion(query, model)
            if added_vocab==-1 or added_vocab==None:
                docs,words_used = gr.simple_results(query)
            else:
                docs,words_used = gr.simple_results(query,added_vocab)

            docs = gr.retrieve_results(docs,words_used)
        else:
            docs,words_used = gr.simple_results(self.entry.get_text())
            docs = gr.retrieve_results(docs,words_used)
        stringy = ''

        if self.check2.get_active():
            docs = Bloomify(docs,words_used).enabled(True)

        if docs==[] or docs == -1:
            stringy = 'No results found'
            x = DialogExample(self,stringy)
            x.run()
            x.destroy()
        else:
            for doc in range(len(docs[:10])):
                stringy = str(doc+1) + ". " + open("corpora/reuters/test/"+docs[doc]).readline()
                self.arx[doc].set_label(stringy)
                self.arx[doc].dialog(self, open("corpora/reuters/test/"+docs[doc]).read())
                self.arx[doc].connect("clicked", self.arx[doc].dialogrun)

    def on_open_clicked(self, button):
        x = DialogExample(self, "It may take upto 2 hours\n\n This is not required if the existing .pkl files provided are used.\n Press OK to continue")
        resp = x.run()
        x.destroy()
        if resp == Gtk.ResponseType.OK:
            idf.get_idf()
            invi.get_inverted_tfidf()
            x = DialogExample(self, "Dictionary successfuly generated")
            x.run()
            x.destroy()


    def on_close_clicked(self, button):
        Gtk.main_quit()

class TitleButton(Gtk.Button):
    def dialog(self, parent, labelval):
        self.parent = parent
        self.labelval = labelval

    def dialogrun(self, button):
        self.dialog = DialogExample(self.parent, self.labelval)
        self.dialog.run()
        self.dialog.destroy()

class DialogExample(Gtk.Dialog):

    def __init__(self, parent, labelval):
        Gtk.Dialog.__init__(self, "My Dialog", parent, 0,
            (Gtk.STOCK_OK, Gtk.ResponseType.OK))

        self.set_default_size(150, 100)

        label = Gtk.Label(labelval)

        box = self.get_content_area()
        box.add(label)
        self.show_all()


class Bloomify(object):
    def __init__(self, docs, words_used):
        self.docs = docs
        self.words_used = words_used
        self.is_enabled = False

    def enabled(self, boole):
        self.is_enabled = boole
        return gr.retrieve_results(self.docs, self.words_used, bloom = True)

print("Loading Glove Vector Model of 5GB. This may take a while. Please wait")
# model = gr.loadGloveModel("GloVe/glove.42B.300d.txt")
win = ButtonWindow()
win.resize(800,600)
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
