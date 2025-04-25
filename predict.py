import argparse
import pickle
import torch
import numpy as np
from preprocess import read_file, loadGloveModel
from model import DocumentGraph
from utils import Data
from sklearn.utils import class_weight

# Load the saved model
def load_model(model_path, args, vocab_size, num_categories, pre_trained_weight, class_weights):
    # Initialize the model with the same architecture as during training
    model = DocumentGraph(args, pre_trained_weight, class_weights, vocab_size, num_categories)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input article
def preprocess_article(article, vocab_dic, max_num_sentence, keywords_dic, num_categories, use_LDA=False):
    # Tokenize the article (you may need to customize this based on your preprocessing)
    tokens = article.lower().split()  # Simple tokenization by splitting on spaces
    indices = [vocab_dic.get(token, 0) for token in tokens]  # Convert tokens to indices, using 0 for unknown words
    
    # Debug: Print tokens and indices
    print("Tokens:", tokens)
    print("Indices:", indices)
    
    # Pad or truncate to match the max_num_sentence
    if len(indices) < max_num_sentence:
        indices += [0] * (max_num_sentence - len(indices))  # Pad with zeros
    else:
        indices = indices[:max_num_sentence]  # Truncate
    
    # Convert to a tensor and add batch dimension
    input_tensor = torch.tensor([indices], dtype=torch.long)
    
    # Debug: Print input tensor shape
    print("Input tensor shape:", input_tensor.shape)
    
    # Generate HT (hypergraph or other required input)
    # You may need to customize this based on how HT is generated during training
    HT = torch.zeros(1, max_num_sentence, max_num_sentence)  # Example placeholder
    
    # Debug: Print HT shape
    print("HT shape:", HT.shape)
    
    return input_tensor, HT

# Predict the category of an article
def predict_article(model, article, vocab_dic, max_num_sentence, keywords_dic, num_categories, use_LDA=False):
    # Preprocess the article
    input_tensor, HT = preprocess_article(article, vocab_dic, max_num_sentence, keywords_dic, num_categories, use_LDA)
    
    # Move input tensor and HT to GPU if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        HT = HT.cuda()
    
    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor, HT)  # Shape: [batch_size, num_categories]
        
        # Debug: Print the output shape
        print("Model output shape:", output.shape)
        
        # Ensure the output has the correct shape [batch_size, num_classes]
        if output.shape[1] != num_categories:
            raise RuntimeError(f"Model output has {output.shape[1]} features, but expected {num_categories} (number of categories).")
        
        # Get the predicted class
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class

# Main function
def main():
    # Define the same arguments as used during training
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='R52', help='dataset name')
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
    parser.add_argument('--initialFeatureSize', type=int, default=300, help='initial size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-6, help='l2 penalty')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
    parser.add_argument('--rand', type=int, default=1234, help='rand_seed')
    parser.add_argument('--normalization', action='store_true', help='add a normalization layer to the end')
    parser.add_argument('--use_LDA', action='store_true', help='use LDA to construct semantic hyperedge')
    args = parser.parse_args()

    # Load the vocabulary and other necessary data
    doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, max_num_sentence, keywords_dic, class_weights = read_file(args.dataset, args.use_LDA)
    
    # Load pre-trained GloVe embeddings if needed
    pre_trained_weight = []
    if args.dataset == 'mr':
        gloveFile = 'data/glove.6B.300d.txt'
        pre_trained_weight = loadGloveModel(gloveFile, vocab_dic, len(vocab_dic) + 1)
    
    # Load the saved model
    model_path = 'hypergat_model.pth'
    model = load_model(model_path, args, len(vocab_dic) + 1, len(labels_dic), pre_trained_weight, class_weights)
    
    # Move the model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    
    # Get user input
    article = input("Enter the article to classify: ")
    
    # Predict the category
    predicted_class = predict_article(model, article, vocab_dic, max_num_sentence, keywords_dic, len(labels_dic), args.use_LDA)
    
    # Map the predicted class index to the actual category name
    category = list(labels_dic.keys())[list(labels_dic.values()).index(predicted_class)]
    print(f"Predicted category: {category}")

if __name__ == '__main__':
    main()