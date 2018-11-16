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
flag=0

class ButtonWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Button Demo")
        self.set_border_width(10)

        vbox = Gtk.Box(spacing=4, orientation = Gtk.Orientation.VERTICAL)
        self.add(vbox)

        hbox_top = Gtk.Box(spacing = 10)
        hbox_mid = Gtk.Box(spacing = 20)
        hbox_bottom = Gtk.Box(spacing = 4)
        vbox.pack_start(hbox_top, False, True, 0)
        vbox.pack_start(hbox_mid, True, True, 0)
        vbox.pack_start(hbox_bottom, False, True, 0)

        self.entry = Gtk.Entry()
        self.entry.set_size_request(250, 30)
        self.entry.set_placeholder_text("Enter the search query here: ")

        button1 = Gtk.Button.new_with_label("Search")
        button1.connect("clicked", self.on_click_me_clicked)

        self.empty_label = Gtk.Label("\n\n\n\n\n\n\n\n\n\n\n\n")

        button2 = Gtk.Button.new_with_mnemonic("Generate Dict")
        button2.connect("clicked", self.on_open_clicked)
        
        button3 = Gtk.Button.new_with_mnemonic("_Close")
        button3.connect("clicked", self.on_close_clicked)

        button4 = Gtk.Button.new_with_mnemonic("Evaluate Results")
        button4.connect("clicked", self.on_click_evaluation)

        self.check1 = Gtk.CheckButton("Embedding Search")
        self.check2 = Gtk.CheckButton("Bloom Filter")

        hbox_top.pack_start(self.entry, False, True, 0)
        hbox_top.pack_end(self.check2, False, True, 0)
        hbox_top.pack_end(self.check1, False, True, 0)
        hbox_top.pack_end(button1, False, True, 0)
        hbox_mid.pack_start(self.empty_label, True, True, 0)
        hbox_bottom.pack_end(button3, False, True, 0)
        hbox_bottom.pack_end(button2, False, True, 0)
        hbox_bottom.pack_end(button4, False, True, 0)


    def on_click_evaluation(self, button):
        self.empty_label.set_text("This may take upto 5 minutes.\n Kindly wait")
        stringy = "Evaluation for normal search\n" + e.normal_evaluation()
        stringy += "\nEvaluation for Embedding search\n" + e.embedding_evaluation(model)
        self.empty_label.set_text(stringy)

    def on_click_me_clicked(self, button):
        global flag
        if self.check1.get_active():
            if flag==0:
                self.empty_label.set_text("May take 5 minutes")
                model = gr.loadGloveModel("GloVe/glove.42B.300d.txt")
            flag=1
            query=str(input())
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
            Bloomify(docs).enabled(True)
    
        if docs==[] or docs == -1:
            stringy = 'No results found'
        else:
            for doc in range(len(docs[:10])):
                f=open("corpora/reuters/test/"+docs[doc])
                stringy += str(doc+1) + ". " + f.readline() + '\n'
        self.empty_label.set_text(stringy)

    def on_open_clicked(self, button):
        self.empty_label.set_text("Started generating dictionary\n It may take upto 2 hours\n\n This is not required if the existing .pkl files provided are used")
        idf.get_idf()
        invi.get_inverted_tfidf()
        empty_label.set_text("Dictionary successfuly generated")

    def on_close_clicked(self, button):
        print("Closing application")
        Gtk.main_quit()


class Bloomify(object):
    def __init__(self, docs):
        self.docs = docs
        self.is_enabled = False

    def enabled(self, boole):
        self.is_enabled = boole
        gr.retrieve_results(docs, bloom = True)

win = ButtonWindow()
win.resize(700,500)
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()