import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
nltk.download('punkt')
import re
from textblob import Word
import plotly.graph_objects as go
from textblob.classifiers import NaiveBayesClassifier
import plotly.express as px

#---------------------------------------------------------------------------------------------

sid = SentimentIntensityAnalyzer()
diary = ''
emot_data = {}
emot_data2 = {}
intensity_data = pd.read_csv('EmotionalIntensity.txt', delim_whitespace=True)
lex_data = pd.read_csv('EmotionalLex.92.txt', delim_whitespace=True)

def clean_words(raw_diary):
    diary = TextBlob(raw_diary)
    stop = set(stopwords.words('english'))
    cleaned_diary = []
    for word in diary.words:
        if word.lower() not in stop:
            cleaned_diary.append(re.sub(r'\W+', '', word.lower()).title())
    return TextBlob(' '.join(cleaned_diary))

def get_wf(diary):
    wf = {}
    length = len(diary.words)
    for word in diary.words:
        wf[word] = wf[word]+1 if word in wf else 1.0
    for word in wf:
        wf[word] = (wf[word], round(wf[word]/length*100,5))
    return wf

#---------------------------------------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H1('E-Diary', style={'text-align': 'center', 'color':'white', 'padding':'1%'}),
    html.Div(dcc.Textarea(id='input-on-submit', value='', style={'padding-top':'2%', 'width': '80%', 'height': 550, 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'})),
    html.Button('Submit', id='submit-val', n_clicks=0, style={'display': 'block', 'margin-left': '83%', 'margin-right': 'auto', 'margin-top':'1%', 'color':'white'}),
    html.H1(children="Statistics", style={"padding":"5%", 'text-align': 'center', 'color':'white'}),
    html.Div(id='container-button-basic',children='Enter a value and press submit', style={'background-color':'white', 'margin':'5%'}),
    html.Div(id='container-button-basic2',children='Enter a value and press submit', style={'background-color':'white', 'margin':'5%'}),
    html.Div(id='container-button-basic3',children='Enter a value and press submit',style={'background-color':'white', 'margin':'5%'}),
    html.Div(id='container-button-basic4',children='Enter a value and press submit', style={'background-color':'white', 'margin':'5%'}),
    html.Div(id='container-button-basic5',children='Enter a value and press submit',style={'background-color':'white', 'margin':'5%'}),
    html.Div(id='container-button-basic6',children='Enter a value and press submit', style={'background-color':'white', 'margin':'5%'}),
    html.Div(id='container-button-basic8',children='Enter a value and press submit',style={'background-color':'white', 'margin':'5%'}),
    html.Div(id='container-button-basic7',children='Enter a value and press submit', style={'background-color':'white', 'margin':'5%'}),
    html.Div(id='container-button-basic9',children='Enter a value and press submit',style={'background-color':'white', 'margin':'5%'}),
    ], style={"background-color":"#3282b8",'margin': '0'})


@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    diary = value
    diary = clean_words(diary)
    text = str(diary)
    text = re.sub(r'\b\w{0,1}\b', '', text)
    diary = TextBlob(text)
    new_diary=[]
    for emotion in intensity_data.emotion.unique():
        emot_data[emotion] = 0.0
    for word in diary.words:
        new_diary.append(word.lower())
    for word in intensity_data.word:
        if word in new_diary:
            for emotion in emot_data.keys():
                try:
                    val = intensity_data.loc[(intensity_data.word==word) & (intensity_data.emotion==emotion)].score.tail(1).iloc[0]
                    emot_data[emotion] += val
                    emot_data[emotion] /= 2
                except:
                    val = 0
    
    
    return html.Div(children=[ dcc.Graph(id='example',
        figure={
            'data': [
                {'x': list(emot_data.keys()), 'y': list(emot_data.values()), 'type': 'bar', 'name': 'Emotions'}
            ],
            'layout': {
                'title': 'Emotions Scale'
            }
        }
    )
   
])
    
@app.callback(
    dash.dependencies.Output('container-button-basic2', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    diary = value
    diary = clean_words(diary)
    text = str(diary)
    text = re.sub(r'\b\w{0,1}\b', '', text)
    diary = TextBlob(text)
    new_diary=[]
    emotions2 = {}
    for emotion in lex_data.emotion.unique():
        emot_data2[emotion] = 0.0
    for word in diary.words:
        new_diary.append(word.lower())
    for word in lex_data.word:
        if word in new_diary:
            for emotion in emot_data2.keys():
                val = lex_data.loc[(lex_data.word==word) & (lex_data.emotion==emotion)].presence.tail(1).iloc[0]
                if val == 1:
                    emotions2[word]=emotion
                    emot_data2[emotion] += 1
                
                
    
    return html.Div(children=[ dcc.Graph(id='example',
        figure={
            'data': [
                {'x': list(emot_data2.keys()), 'y': list(emot_data2.values()), 'type': 'bar', 'name': 'emotions'}
            ],
            'layout': {
                'title': 'Emotional Intensity'
            }
        }
    )
   
])
    
@app.callback(
    dash.dependencies.Output('container-button-basic3', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    diary = value
    diary = TextBlob(diary)
    sentence_polarity = []
    sentence_subjectivity = []
    for sentence in diary.sentences:
        sentence_polarity.append(sentence.sentiment.polarity)
        sentence_subjectivity.append(sentence.sentiment.subjectivity)
    sen = []
    for sentence in diary.sentences:
        sen.append(str(sentence))
    sen = []
    for sentence in diary.sentences:
        sen.append(str(sentence))
    fig = go.Figure(data=[go.Table(header=dict(values=['Sentence', 'Polarity', 'Subjectivity']),
                 cells=dict(values=[sen, sentence_polarity, sentence_subjectivity]))
                     ])
    return dcc.Graph(figure=fig)

@app.callback(
    dash.dependencies.Output('container-button-basic4', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    diary = value
    diary = TextBlob(diary)
    sentence_polarity = []
    sentence_subjectivity = []
    for sentence in diary.sentences:
        sentence_polarity.append(sentence.sentiment.polarity)
        sentence_subjectivity.append(sentence.sentiment.subjectivity)
    sen = []
    for sentence in diary.sentences:
        sen.append(str(sentence))
    x=[]
    for i in range(len(sen)):
        x.append(i)
    return html.Div(children=[ dcc.Graph(id='example',
        figure={
            'data': [
                {'x': x, 'y': sentence_polarity, 'type': 'line', 'name': 'emotional stability'}
            ],
            'layout': {
                'title': 'Emotions throughout Diary Entry'
            }
        }
    )
   
])
    
@app.callback(
    dash.dependencies.Output('container-button-basic5', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    diary = value
    diary = TextBlob(diary)
    sentence_polarity = []
    sentence_subjectivity = []
    for sentence in diary.sentences:
        sentence_polarity.append(sentence.sentiment.polarity)
        sentence_subjectivity.append(sentence.sentiment.subjectivity)
    sen = []
    for sentence in diary.sentences:
        sen.append(str(sentence))
    x=[]
    for i in range(len(sen)):
        x.append(i)
    return html.Div(children=[ dcc.Graph(id='example',
        figure={
            'data': [
                {'x': x, 'y': sentence_subjectivity, 'type': 'line', 'name': 'subjectivity'}
            ],
            'layout': {
                'title': 'Phase Generality'
            }
        }
    )
   
])
        

@app.callback(
    dash.dependencies.Output('container-button-basic6', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    diary = value
    diary = TextBlob(diary)
    train = [
         ('I love this sandwich.', 'pos'),
         ('this is an amazing place!', 'pos'),
         ('I feel very good about these beers.', 'pos'),
         ('this is my best work.', 'pos'),
         ("what an awesome view", 'pos'),
         ('I do not like this restaurant', 'neg'),
         ('I am tired of this stuff.', 'neg'),
         ("I can't deal with this", 'neg'),
         ('he is my sworn enemy!', 'neg'),
         ('my boss is horrible.', 'neg'),
     ]
    test = [
         ('the beer was good.', 'pos'),
         ('I do not enjoy my job', 'neg'),
         ("I ain't feeling dandy today.", 'neg'),
         ("I feel amazing!", 'pos'),
         ('Gary is a friend of mine.', 'pos'),
         ("I can't believe I'm doing this.", 'neg')
     ]
    classifier = NaiveBayesClassifier(train)
    sen = []
    for sentence in diary.sentences:
        sen.append(str(sentence))
    class_sentence_list = []
    for sentence in diary.sentences:
        class_sentence_list.append(classifier.classify(sentence))
    fig = go.Figure(data=[go.Table(header=dict(values=['Sentence', 'Negative/Positive']),
                 cells=dict(values=[sen, class_sentence_list]))
                     ])
    return dcc.Graph(figure=fig)


@app.callback(
    dash.dependencies.Output('container-button-basic7', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    diary = value
    diary = clean_words(diary)
    text = str(diary)
    text = re.sub(r'\b\w{0,1}\b', '', text)
    diary_for_freq = TextBlob(text)
    word_freq = get_wf(diary_for_freq)
    word_freq_list = list(word_freq.keys())
    word_freq_it = list(word_freq.values())
    word_x = []
    word_y = []
    for w in word_freq_it:
        word_x.append(w[1])
        word_y.append(w[0])
    fig = go.Figure(data=[go.Table(header=dict(values=['Word', 'Frequency','Percentage']),
                 cells=dict(values=[word_freq_list, word_y, word_x]))
                     ])
    return dcc.Graph(figure=fig)
    

@app.callback(
    dash.dependencies.Output('container-button-basic8', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    diary = value
    diary = clean_words(diary)
    text = str(diary)
    text = re.sub(r'\b\w{0,1}\b', '', text)
    diary_for_freq = TextBlob(text)
    word_freq = get_wf(diary_for_freq)
    word_freq_list = list(word_freq.keys())
    word_freq_it = list(word_freq.values())
    word_x = []
    for w in word_freq_it:
        word_x.append(w[1])
    data = {'a': word_freq_list, 'b':word_x}
    df = pd.DataFrame.from_dict(data)
    fig = px.pie(df, values='b', names='a')
    return dcc.Graph(figure=fig)


@app.callback(
    dash.dependencies.Output('container-button-basic9', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    diary = value
    diary_dict = sid.polarity_scores(diary)
    x = []
    y = []
    for d in diary_dict:
        x.append(d)
        y.append(diary_dict[d])
    diary_dict = { 'X':x, 'Y':y}
    return html.Div(children=[ dcc.Graph(id='example',
        figure={
            'data': [
                {'x': x, 'y': y, 'type': 'bar', 'name': 'genre'}
            ],
            'layout': {
                'title': 'Overall Emotion'
            }
        }
    )
   
])
    
    

        
   



if __name__ == '__main__':
    app.run_server(debug=True)