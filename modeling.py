import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AlbertPreTrainedModel, AlbertModel, DistilBertPreTrainedModel, DistilBertModel

class BertForSentimentClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.bert = BertModel(config)
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		#The classification layer that takes the [CLS] representation and outputs the logit
		self.cls_layer = nn.Linear(config.hidden_size, 1)

	def forward(self, input_ids, attention_mask):
		'''
		Inputs:
			-input_ids : Tensor of shape [B, T] containing token ids of sequences
			-attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the input length)
		'''
		#Feed the input to Bert model to obtain outputs
		outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		#Obtain the representations of [CLS] heads
		cls_reps = outputs[1]
		# cls_reps = self.dropout(cls_reps)
		logits = self.cls_layer(cls_reps)
		return logits

class AlbertForSentimentClassification(AlbertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.albert = AlbertModel(config)
		#The classification layer that takes the [CLS] representation and outputs the logit
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, input_ids, attention_mask):
		'''
		Inputs:
			-input_ids : Tensor of shape [B, T] containing token ids of sequences
			-attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the input length)
		'''
		#Feed the input to Albert model to obtain outputs
		outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
		#Obtain the representations of [CLS] heads
		cls_reps = outputs.last_hidden_state[:, 0]
		logits = self.cls_layer(cls_reps)
		return logits

class DistilBertForSentimentClassification(DistilBertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.distilbert = DistilBertModel(config)
		#The classification layer that takes the [CLS] representation and outputs the logit
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, input_ids, attention_mask):
		'''
		Inputs:
			-input_ids : Tensor of shape [B, T] containing token ids of sequences
			-attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the input length)
		'''
		#Feed the input to DistilBert model to obtain outputs
		outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
		#Obtain the representations of [CLS] heads
		cls_reps = outputs.last_hidden_state[:, 0]
		logits = self.cls_layer(cls_reps)
		return logits


# class BertEmbeddings(nn.Module):
#     """Construct the embeddings from word, position and token_type embeddings.
#     """
#     def __init__(self, config):
#         super(BertEmbeddings, self).__init__()
#
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
#         self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)
#
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#     def forward(self, input_ids, token_type_ids=None):
#         seq_length = input_ids.size(1)
#
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)
#
#         words_embeddings = self.word_embeddings(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)
#
#         embeddings = words_embeddings + token_type_embeddings + position_embeddings #wiq_embeddings + position_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#
#         return embeddings