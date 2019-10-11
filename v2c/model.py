import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# Functions for V2CNet
# ----------------------------------------

class VideoEncoder(nn.Module):
    """Module to encode frame image from pre-trained CNN.
    """
    def __init__(self,
                 in_size,
                 units):
        super(VideoEncoder, self).__init__()
        self.linear = nn.Linear(in_size, units)
        self.lstm = nn.LSTM(units, units, batch_first=True)

    def forward(self, 
                Xv):
        # Phase 1: Encoding Stage
        # Encode video features with one dense layer and lstm
        # State of this lstm to be used for lstm2 language generator
        Xv = self.linear(Xv)
        #print('linear:', Xv.shape)
        Xv = F.relu(Xv)

        Xv, (hi, ci) = self.lstm(Xv)
        Xv = Xv[:,-1,:]     # Only need the last timestep
        hi, ci = hi[0,:,:], ci[0,:,:]
        #print('lstm:', Xv.shape, 'hi:', hi.shape, 'ci:', ci.shape)
        return Xv, (hi, ci)


class CommandDecoder(nn.Module):
    """Module to decode features and generate word for captions
    using RNN.
    """
    def __init__(self,
                 units,
                 vocab_size,
                 embed_dim,
                 bias_init_vector=None):
        super(CommandDecoder, self).__init__()
        self.units = units
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim, units)
        self.logits = nn.Linear(units, vocab_size, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        if bias_init_vector is not None:
            self.logits.bias.data = torch.from_numpy(bias_init_vector).float()

    def forward(self, 
                Xs, 
                states):
        # Phase 2: Decoding Stage
        # Given the previous word token, generate next caption word using lstm2
        # Sequence processing and generating
        #print('sentence decoding stage:')
        #print('Xs:', Xs.shape)
        Xs = self.embed(Xs)
        #print('embed:', Xs.shape)

        hi, ci = self.lstm_cell(Xs, states)
        #print('out:', hi.shape, 'hi:', states[0].shape, 'ci:', states[1].shape)

        x = self.logits(hi)
        #print('logits:', x.shape)
        x = self.softmax(x)
        #print('softmax:', x.shape)
        return x, (hi, ci)

    def init_hidden(self, batch_size):
        """Initialize a zero state for LSTM.
        """
        h0 = torch.zeros(batch_size, self.units)
        c0 = torch.zeros(batch_size, self.units)
        return (h0, c0)


class CommandLoss(nn.Module):
    """Calculate Cross-entropy loss per word.
    """
    def __init__(self, 
                 ignore_index=0):
        super(CommandLoss, self).__init__()
        self.cross_entropy = nn.NLLLoss(reduction='sum', 
                                        ignore_index=ignore_index)

    def forward(self, 
                input, 
                target):
        return self.cross_entropy(input, target)


class Video2Command():
    """Train/Eval inference class for V2C model.
    """
    def __init__(self,
                 config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def build(self,
              optimizer):
        # Initialize Encode & Decode models here
        self.video_encoder = VideoEncoder(in_size=self.config.NUM_FEATURES, 
                                          units=self.config.UNITS)
        self.command_decoder = CommandDecoder(units=self.config.UNITS,
                                              vocab_size=self.config.VOCAB_SIZE,
                                              embed_dim=self.config.EMBED_SIZE)
        # Loss function
        self.loss_objective = CommandLoss()

        # Setup parameters and optimizer
        self.params = list(self.video_encoder.parameters()) + \
                      list(self.command_decoder.parameters())
        self.optimizer = torch.optim.Adam(self.params, 
                                          lr=self.config.LEARNING_RATE)

        # Save configuration
        # Safely create checkpoint dir if non-exist
        if not os.path.exists(os.path.join(self.config.CHECKPOINT_PATH, 'saved')):
            os.makedirs(os.path.join(self.config.CHECKPOINT_PATH, 'saved'))

    def train(self, 
              train_loader):
        """Train the Video2Command model.
        """
        def train_step(Xv, S):
            """One train step.
            """
            loss = 0.0
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # Video feature extraction 1st
            Xv, states = self.video_encoder(Xv)

            # Calculate mask against zero-padding
            S_mask = S != 0

            # Teacher-Forcing for command decoder
            for timestep in range(self.config.MAXLEN - 1):
                Xs = S[:,timestep]
                probs, states = self.command_decoder(Xs, states)
                # Calculate loss per word
                loss += self.loss_objective(S[:,timestep+1], probs)
            loss = loss / S_mask.sum()     # Loss per word

            # Gradient backward
            loss.backward()
            self.optimizer.step()
            return loss

        # Training epochs
        self.video_encoder.train()
        self.command_decoder.train()
        for epoch in range(self.config.NUM_EPOCHS):
            total_loss = 0.0
            for i, (Xv, S) in enumerate(train_loader):
                # Mini-batch
                Xv, S = Xv.to(self.device), S.to(self.device)
                loss = train_step(Xv, S)
                total_loss += loss
                # Display
                if i % self.config.DISPLAY_EVERY == 0:
                    print('Epoch {}, Iter {}, Loss {:.6f}'.format(epoch+1, 
                                                                  i,
                                                                  loss))
            # End of epoch, save weights
            print('Total loss for epoch {}: {:.6f}'.format(epoch+1, total_loss / (i + 1)))
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_weights()
        return

    def evaluate(self,
                 eval_loader,
                 vocab):
        """Run the evaluation pipeline over the test dataset.
        """
        assert self.config.MODE == 'eval'
        y_pred, y_true = [], []
        # Evaluation over the entire test dataset
        for i, (Xv, S_true) in enumerate(test_loader):
            S_pred = self.predict(Xv, vocab)
            y_pred.append(S_pred)
            y_true.append(S_true)
        y_pred = torch.concat(y_pred, axis=0)
        y_true = torch.concat(y_true, axis=0)
        return y_pred.numpy(), y_true.numpy()

    def predict(self, 
                Xv,
                vocab):
        """Run the prediction pipeline given one sample.
        """
        self.video_encoder.eval()
        self.command_decoder.eval()

        # Initialize S with '<sos>'
        S = torch.zeros((Xv.shape[0], self.config.MAXLEN), dtype=torch.long)
        S[:,0] = vocab('<sos>')

        # Start v2c prediction pipeline
        Xv, states = self.video_encoder(Xv)

        #states = self.command_decoder.reset_states(Xv.shape[0])
        #_, states = self.command_decoder(None, states, Xv=Xv)   # Encode video features 1st
        for timestep in range(self.config.MAXLEN - 1):
            Xs = S[:,timestep]
            probs, states = self.command_decoder(Xs, states)
            preds = torch.argmax(probs, dim=1)    # Collect prediction
            S[:,timestep+1] = preds
        return S

    def save_weights(self):
        """Save the current weights and record current training info 
        into tensorboard.
        """
        # Save the current checkpoint
        torch.save({
                    'VideoEncoder_state_dict': self.video_encoder.state_dict(),
                    'CommandDecoder_state_dict': self.command_decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(self.config.CHECKPOINT_PATH, 'saved'))
        print('Model saved.')

    def load_weights(self,
                     save_path):
        """Load pre-trained weights by path.
        """
        print('Loading...')
        checkpoint = torch.load(save_path)
        self.video_encoder.load_state_dict(checkpoint['VideoEncoder_state_dict'])
        self.command_decoder.load_state_dict(checkpoint['CommandDecoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Model loaded.')