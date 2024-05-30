import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import models
from v2c.backbone import resnet
import sys

# ----------------------------------------
# Functions for Video Feature Extraction
# ----------------------------------------

class CNNWrapper(nn.Module):
    """Wrapper module to extract features from image using
    pre-trained CNN.
    """
    def __init__(self,
                 backbone,
                 checkpoint_path):
        super(CNNWrapper, self).__init__()
        self.backbone = backbone
        self.model = self.init_backbone(checkpoint_path)

    def forward(self,
                x):
        with torch.no_grad():
            x = self.model(x)
        x = x.reshape(x.size(0), -1)
        return x

    def init_backbone(self,
                      checkpoint_path):
        """Helper to initialize a pretrained pytorch model.
        """
        if self.backbone == 'resnet50':
            model = resnet.resnet50(pretrained=False)   # Use Caffe ResNet instead
            model.load_state_dict(torch.load(checkpoint_path))

        elif self.backbone == 'resnet101':
            model = resnet.resnet101(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))

        elif self.backbone == 'resnet152':
            model = resnet.resnet152(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))

        elif self.backbone == 'vgg16':
            model = models.vgg16(pretrained=True)

        elif self.backbone == 'vgg19':
            model = models.vgg19(pretrained=True)

        # Remove the last classifier layer
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
        
        return model

# ----------------------------------------
# Transformer Encoder and Decoder for V2CNet
# ----------------------------------------
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, 
                 dropout: float = 0.1, 
                 max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, config, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.positional_encoder = PositionalEncoding(d_model=config.N_FEATURES, 
                                                     dropout=dropout, 
                                                     max_len=config.WINDOW_SIZE)
        self.encoder = nn.TransformerEncoder(
                                nn.TransformerEncoderLayer( config.N_FEATURES, 
                                                            config.N_HEAD, 
                                                            dim_feedforward=dim_feedforward, 
                                                            dropout=dropout), config.N_ENCODING_LAYER)

        self.linear = nn.Linear(config.N_FEATURES, config.EMBED_SIZE)

    def forward(self, x):
        # Encode video features with Transformer
        x = self.positional_encoder(x)
        x = self.encoder(x)
        return self.linear(x)

class TransformerDecoder(nn.Module):
    def __init__(self, config, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBED_SIZE)
        self.position_embedding = PositionalEncoding(d_model=config.EMBED_SIZE,
                                                        dropout=dropout,
                                                        max_len=config.MAXLEN)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(config.EMBED_SIZE, 
                                                                            config.N_HEAD, 
                                                                            dim_feedforward, 
                                                                            dropout), config.N_DECODING_LAYER)

    def forward(self, target, enc_out, target_mask, target_mask_padding):
        # Decode features and generate captions using Transformer
        x = self.embedding(target)
        x = self.position_embedding(x)
        output = self.decoder(x, enc_out, tgt_mask=target_mask, tgt_key_padding_mask=target_mask_padding)
        return output

class TransformerV2C(nn.Module):
    def __init__(self, config, d_hid: int = 2048, dropout: float = 0.5):
        super().__init__()
        
        self.encoder = TransformerEncoder(config, d_hid, dropout)
        self.decoder = TransformerDecoder(config, d_hid, dropout)
        self.linear = nn.Linear(config.EMBED_SIZE, config.VOCAB_SIZE)
  
    def forward(self, v_feat, tgt, tgt_mask, tgt_mask_padding):
        enc_out = self.encoder(v_feat)
        output = self.decoder(tgt, enc_out, tgt_mask, tgt_mask_padding)
        output = self.linear(output)
        return output

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(tgt):
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, tgt.device)

    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return tgt_mask, tgt_padding_mask

# ----------------------------------------
# Functions for V2CNet
# ----------------------------------------
class CommandLoss(nn.Module):
    """Calculate Cross-entropy loss per word.
    """
    def __init__(self, 
                 ignore_index=0):
        super(CommandLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum', 
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
              bias_vector=None):
        # Initialize Encode & Decode models here
        self.transformerV2C = TransformerV2C(self.config, d_hid=2048, dropout=0.5)
        self.transformerV2C.to(self.device)
    
        # Loss function
        self.loss_objective = CommandLoss()
        self.loss_objective.to(self.device)

        # Setup parameters and optimizer
        self.params = list(self.transformerV2C.parameters())
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
            self.optimizer.zero_grad()

            Xv = Xv.permute(1, 0, 2)   # (B, L, C) -> (L, B, C)
            S = S.permute(1, 0)   # (B, L) -> (L, B)

            tgt_input = S[:-1, :]
            tgt_output = S[1:, :]

            # Get target_mask for Transformer
            tgt_mask, tgt_padding_mask = create_mask(tgt_input)

            # Get captions with Transformer
            logits = self.transformerV2C(Xv, tgt_input, 
                                         tgt_mask=tgt_mask, 
                                         tgt_mask_padding=tgt_padding_mask)
            loss = self.loss_objective(logits.reshape(-1, logits.shape[-1]), 
                                       tgt_output.reshape(-1))

            # loss = loss / S_mask.sum()     # Loss per word

            # Gradient backward
            loss.backward()
            self.optimizer.step()
            return loss

        # Training epochs
        self.transformerV2C.train()
        for epoch in range(self.config.NUM_EPOCHS):
            total_loss = 0.0
            for i, (Xv, S, clip_names) in enumerate(train_loader):
                # Mini-batch
                Xv, S = Xv.to(self.device), S.to(self.device)

                # Train step
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
                self.save_weights(epoch + 1)
        return

    def evaluate(self,
                 test_loader,
                 vocab):
        """Run the evaluation pipeline over the test dataset.
        """
        assert self.config.MODE == 'test'
        y_pred, y_true = [], []
        # Evaluation over the entire test dataset
        for i, (Xv, S_true, clip_names) in enumerate(test_loader):
            # Mini-batch
            Xv, S_true = Xv.to(self.device), S_true.to(self.device)
            S_pred = self.predict(Xv, vocab)
            y_pred.append(S_pred)
            y_true.append(S_true)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        return y_pred.cpu().numpy(), y_true.cpu().numpy()

    def predict(self, 
                Xv,
                vocab):
        """Run the prediction pipeline given one sample.
        """
        self.video_encoder.eval()
        self.command_decoder.eval()

        with torch.no_grad():
            # Initialize S with '<sos>'
            S = torch.zeros((Xv.shape[0], self.config.MAXLEN), dtype=torch.long)
            S[:,0] = vocab('<sos>')
            S = S.to(self.device)

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

    def save_weights(self, 
                     epoch):
        """Save the current weights and record current training info 
        into tensorboard.
        """
        # Save the current checkpoint
        torch.save({
                    'transformerV2C_state_dict': self.transformerV2C.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(self.config.CHECKPOINT_PATH, 'saved', 'v2c_epoch_{}.pth'.format(epoch)))
        print('Model saved.')

    def load_weights(self,
                     save_path):
        """Load pre-trained weights by path.
        """
        print('Loading...')
        checkpoint = torch.load(save_path)
        self.transformerV2C.load_state_dict(checkpoint['transformerV2C_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Model loaded.')
