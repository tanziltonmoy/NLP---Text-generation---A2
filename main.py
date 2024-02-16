import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pickle
import torch
import torch.nn as nn
import math 
from torchtext.data.utils import get_tokenizer

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim    = hid_dim
        self.emb_dim    = emb_dim
        
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hid_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim,
                self.hid_dim).uniform_(-init_range_other, init_range_other) #We
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim,   
                self.hid_dim).uniform_(-init_range_other, init_range_other) #Wh
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
        
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach() #not to be used for gradient computation
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden):
        #src: [batch_size, seq len]
        embedding = self.dropout(self.embedding(src)) #harry potter is
        #embedding: [batch-size, seq len, emb dim]
        output, hidden = self.lstm(embedding, hidden)
        #ouput: [batch size, seq len, hid dim]
        #hidden: [num_layers * direction, seq len, hid_dim]
        output = self.dropout(output)
        prediction =self.fc(output)
        #prediction: [batch_size, seq_len, vocab_size]
        return prediction, hidden


def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens
   
# Load the model and tokenizer
with open('model.pkl', 'rb') as data_file:
    lstm = pickle.load(data_file)

# Extracting data for the language model
vocab_size = lstm['vocab_size']
emb_dim = lstm['emb_dim']
hid_dim = lstm['hid_dim']
num_layers = lstm['num_layers']
dropout_rate = lstm['dropout_rate']
tokenizer = lstm['tokenizer']
vocab = lstm['vocab']

# Instantiate the model
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
model.load_state_dict(torch.load('best-val-lstm_lm.pt', map_location=torch.device('cpu')))
model.eval()

# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout with Bootstrap components
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1('Harry Potter Text Generator'), width={"size": 6, "offset": 3}, className="mb-4 mt-4")),
    dbc.Row([
        dbc.Col(dcc.Input(id='user-input', type='text', placeholder='Enter text...', className="mb-3"), width=10),
        dbc.Col(html.Button('Generate', id='submit-val', n_clicks=0, className="btn btn-primary mb-3"), width=2),
    ]),
    dbc.Row(dbc.Col(html.Div(id='autocomplete-result'), width=12))
], fluid=True)

@app.callback(
    Output('autocomplete-result', 'children'),
    [Input('submit-val', 'n_clicks')],
    [State('user-input', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        prompt = value
        seq_len = 30
        temperature = 0.5
        seed = 0
        device = torch.device('cpu')  # Assuming CPU usage; adjust as necessary
        
        generation = generate(prompt, seq_len, temperature, model, tokenizer, vocab, device, seed)
        sentence = ' '.join(generation)
        return dbc.Alert(f"Generated Text: {sentence}", color="success")
    else:
        return dbc.Alert('Enter text and click generate to see the autocomplete.', color="secondary")

if __name__ == '__main__':
    app.run_server(debug=True)