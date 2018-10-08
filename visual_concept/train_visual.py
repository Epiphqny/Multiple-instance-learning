import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader_v3 import get_loader 
from build_vocab_v3 import Vocabulary
from visual_concept import EncoderCNN, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import time
import random


class Sample_loss(torch.nn.Module):
    def __init__(self):
        super(Sample_loss,self).__init__()

    def forward(self,x,y,lengths):
         loss=0
         batch_size=len(lengths)//8
         for i in range(batch_size):
              label_index=y[i][:lengths[i]]
              values=1-x[i][label_index]
              prod=1
              for value in values:
                  prod=prod*value
              print('prod',prod)
              loss+=1-prod
         loss=Variable(loss, requires_grad=True).unsqueeze(0)
         return loss

class bce_loss(torch.nn.Module):
    def __init__(self):
        super(bce_loss,self).__init__()

    def forward(self,x,y):
         loss=F.binary_cross_entropy(x,y)
         loss=Variable(loss.cuda(), requires_grad=True)
         return loss


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    print("load vocabulary ...")    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print("build data loader ...")
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    
    print("build the models ...")
    # Build the models
    encoder = nn.DataParallel(EncoderCNN()).cuda()
    decoder = nn.DataParallel(Decoder()).cuda()
    
    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    time_start=time.time() 
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, targets,lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images.cuda()
            targets = targets.cuda()
            
            #targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
     
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features)
           
            pos=nn.functional.binary_cross_entropy(outputs*targets,targets)*1e3
            neg=nn.functional.binary_cross_entropy(outputs*(1-targets)+targets,targets)*1
         
            loss=pos+neg
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                time_end=time.time()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time:{}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(),time_end-time_start))
                time_start=time_end
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/zh_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/img_vector.txt', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=1, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=390, help='step size for saving trained models')
    
    # Model parameters
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    print(args)
    main(args)
